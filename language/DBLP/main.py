from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BertForNextSentencePrediction
from datasets import load_dataset
import evaluate

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import torch
import numpy as np

import argparse
import warnings

warnings.simplefilter("ignore")
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data():
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)
    return books


def tokenize_data(examples, args):
    def preprocess_function(examples):
        prefix = "translate English to French: "
        inputs = [prefix + example["en"] for example in examples["translation"]]
        targets = [example["fr"] for example in examples["translation"]]
        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=args.max_sequence_length,
            truncation=True,
        )
        return model_inputs

    tokenized_data = examples.map(preprocess_function, batched=True)
    return tokenized_data


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def get_model(checkpoint):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # model = BertForNextSentencePrediction.from_pretrained(checkpoint)
    # Freeze some layers
    trainable_layers = [
        model.decoder.block[-1].layer[-1].DenseReluDense.wi,
        model.decoder.block[-1].layer[-1].DenseReluDense.wo,
    ]
    total_params = 0
    trainable_params = 0

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

    # Unfreeze some parameters
    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    print(f"Total parameters count: {total_params}")  # 60M
    print(f"Trainable parameters count: {trainable_params}")  # 2M
    return model


def train(args, model, optimizer, train_dataloader, test_dataloader):
    model.train()
    privacy_engine = PrivacyEngine()

    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_delta=args.delta,
        target_epsilon=args.epsilon,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )

    device = args.device
    for epoch in range(1, args.epochs + 1):
        losses = []

        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=args.batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for step, batch in enumerate(memory_safe_data_loader):
                optimizer.zero_grad()

                # batch = tuple(t.to(device) for t in batch)
                # inputs = {"input_ids": batch[0], "labels": batch[1]}
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }

                outputs = model(
                    **inputs
                )  # output = loss, logits, hidden_states, attentions

                loss = outputs[0]
                loss.backward()
                losses.append(loss.item())

                optimizer.step()

                # if step > 0 and step % 100 == 0:
            train_loss = np.mean(losses)
            eps = privacy_engine.get_epsilon(args.delta)

            eval_loss, eval_accuracy = evaluate(args, model, test_dataloader)

            print(
                f"Epoch: {epoch} | "
                # f"Step: {step} | "
                f"Train loss: {train_loss:.3f} | "
                f"Eval loss: {eval_loss:.3f} | "
                f"Eval accuracy: {eval_accuracy:.3f} | "
                f"ɛ: {eps:.2f}"
            )

    # Final evaluation
    _, eval_accuracy = evaluate(args, model, test_dataloader)
    eps = privacy_engine.get_epsilon(args.delta)
    print(
        f"Final evaluation | " f"Eval accuracy: {eval_accuracy:.3f} | " f"ɛ: {eps:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Opacus IMDB trained by Transformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset file management arguments
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Where DBLP is/will be stored"
    )
    # Training arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="t5-small",
        help="Which checkpoint to use",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        metavar="B",
        help="input batch size for test",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        metavar="LR",
        help="learning rate",
    )
    # Differentially private parameters:
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.56,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max_grad_norm",
        type=float,
        default=0.1,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=7.5,
        metavar="D",
        help="Target privacy budget",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=128,
        metavar="SL",
        help="Longer sequences will be cut to this length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    metric = evaluate.load("sacrebleu")

    # Prepare dataset
    examples = load_data()
    tokenized_data = tokenize_data(examples, args)
    # import pdb; pdb.set_trace()
    tokenized_data.set_format(type="torch", columns=["input_ids", "labels"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.checkpoint)
    train_dataset = tokenized_data["train"]
    test_dataset = tokenized_data["test"]

    # import pdb; pdb.set_trace()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # Differentially private training
    model = get_model(args.checkpoint)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    train(args, model, optimizer, train_dataloader, test_dataloader)

    # Train model
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="outputs",
    #     evaluation_strategy="epoch",
    #     learning_rate=args.lr,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     dataloader_num_workers=args.workers,
    #     weight_decay=0.01,
    #     save_total_limit=3,
    #     num_train_epochs=args.epochs,
    #     predict_with_generate=True,
    #     fp16=True,
    #     push_to_hub=False,
    # )

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     optimizers=(optimizer, None),
    #     args=training_args,
    #     train_dataset=tokenized_data["train"],
    #     eval_dataset=tokenized_data["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    # trainer.train()

    # Without DP
    # Result: Only last FF layer of the last block
    # {'eval_loss': 1.91495680809021, 'eval_bleu': 4.1323, 'eval_gen_len': 17.7561, 'eval_runtime': 189.0818, 'eval_samples_per_second': 134.423, 'eval_steps_per_second': 4.205, 'epoch': 1.0}
    # {'eval_loss': 1.8922994136810303, 'eval_bleu': 4.1165, 'eval_gen_len': 17.7505, 'eval_runtime': 189.9937, 'eval_samples_per_second': 133.778, 'eval_steps_per_second': 4.184, 'epoch': 2.0}

    # Result: last block
    # {'eval_loss': 1.836706519126892, 'eval_bleu': 4.4162, 'eval_gen_len': 17.6741, 'eval_runtime': 187.3659, 'eval_samples_per_second': 135.654, 'eval_steps_per_second': 4.243, 'epoch': 1.0}
    # {'eval_loss': 1.815298318862915, 'eval_bleu': 4.3811, 'eval_gen_len': 17.6603, 'eval_runtime': 188.2366, 'eval_samples_per_second': 135.027, 'eval_steps_per_second': 4.223, 'epoch': 2.0}

    # Resuls: full model
    # {'eval_loss': 1.6619088649749756, 'eval_bleu': 5.2619, 'eval_gen_len': 17.6156, 'eval_runtime': 189.3895, 'eval_samples_per_second': 134.205, 'eval_steps_per_second': 4.198, 'epoch': 1.0}
    # {'eval_loss': 1.639931082725525, 'eval_bleu': 5.4179, 'eval_gen_len': 17.6062, 'eval_runtime': 191.0498, 'eval_samples_per_second': 133.039, 'eval_steps_per_second': 4.161, 'epoch': 2.0}{'train_runtime': 1440.1939, 'train_samples_per_second': 141.187, 'train_steps_per_second': 4.413, 'train_loss': 1.9040913182733343, 'epoch': 2.0}
    # Todo: run with Opacus
