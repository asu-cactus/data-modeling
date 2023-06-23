from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset, DatasetDict
import evaluate

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import torch
import numpy as np

import json
import argparse
import warnings

warnings.simplefilter("ignore")
import os
import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "true"

TOPK = 10  # In evaluation, the best bleu score of the topk similar titles is reported


def load_data():
    # Construct train data
    with open("data/train_data.json", "r") as f:
        train_data = json.load(f)

    train_data = [
        {
            "title": example["title"],
            "author&year": f"authors: {example['authors']}, year: {example['year']}",
        }
        for example in train_data["training_articles"]
    ]
    train_data = Dataset.from_list(train_data)
    # Construct test data
    with open("data/test_data.json", "r") as f:
        test_raw = json.load(f)
    # test_raw["articles"] = test_raw["articles"][:100]  # TODO: remove this line
    test_data = []
    for example in test_raw["articles"]:
        title = example["title"]
        for similar_examples in example["topKSimilarArticles"]:
            test_data.append(
                {
                    "title": title,
                    "author&year": f"authors: {similar_examples['authors']}, year: {similar_examples['year']}",
                }
            )
    test_data = Dataset.from_list(test_data)

    # Construct DatasetDict
    data = DatasetDict({"train": train_data, "test": test_data})
    return data

    # books = load_dataset("opus_books", "en-fr")
    # books = books["train"].train_test_split(test_size=0.9)
    # return books


def tokenize_data(examples, args):
    def preprocess_function(samples):
        prefix = "Output authors and publish year for title: "
        inputs = [prefix + example for example in samples["title"]]
        targets = [example for example in samples["author&year"]]
        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=args.max_sequence_length,
            truncation=True,
        )
        return model_inputs

    tokenized_data = examples.map(preprocess_function, batched=True)
    return tokenized_data


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


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result["score"]

    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def eval(args, model, test_dataloader):  # TODO: modify for sequence output
    model.eval()
    metric = evaluate.load("bleu")

    losses = 0
    bleu_scores = []

    device = torch.device(args.device)
    for batch in test_dataloader:
        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            # inputs = {"input_ids": batch[0], "labels": batch[1]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = np.argmax(logits.detach().cpu().numpy(), axis=2)
            labels = batch["labels"].numpy()

            losses += loss.item()
            bleu_scores.append(compute_metrics((preds, labels), metric))

    data_size = len(test_dataloader.dataset)
    bleu_score = (
        sum(
            [
                max(bleu_scores[start : start + TOPK])
                for start in range(0, len(bleu_scores), TOPK)
            ]
        )
        / data_size
        * TOPK
    )
    loss = losses / data_size
    model.train()
    return loss, bleu_score


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

            eval_loss, bleu_score = eval(args, model, test_dataloader)

            print(
                f"Epoch: {epoch} | "
                # f"Step: {step} | "
                f"Train loss: {train_loss:.3f} | "
                f"Eval loss: {eval_loss:.3f} | "
                f"Eval bleu_score: {bleu_score:.3f} | "
                f"ɛ: {eps:.2f}"
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
        default=20,
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
        default=20.0,
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
        default=64,
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

    # Prepare dataset
    examples = load_data()
    tokenized_data = tokenize_data(examples, args)

    tokenized_data.set_format(type="torch", columns=["input_ids", "labels"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.checkpoint)
    train_dataset = tokenized_data["train"]
    test_dataset = tokenized_data["test"]

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
    model = model.to(torch.device(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    train(args, model, optimizer, train_dataloader, test_dataloader)

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
