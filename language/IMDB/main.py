import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForSequenceClassification

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import argparse
import warnings

warnings.simplefilter("ignore")


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


# define evaluation cycle
def evaluate(args, model, test_dataloader):
    model.eval()

    losses = 0
    n_correct = 0

    device = args.device
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            # inputs = {'input_ids':      batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2],
            #           'labels':         batch[3]}
            inputs = {"input_ids": batch[0], "labels": batch[1]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs["labels"].detach().cpu().numpy()

            losses += loss.item()
            n_correct += np.sum(preds == labels)

    data_size = len(test_dataloader.dataset)
    accuracy = n_correct / data_size
    model.train()
    return losses / data_size, accuracy


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

                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "labels": batch[1]}
                # inputs = {'input_ids':      batch[0],
                #         'attention_mask': batch[1],
                #         'token_type_ids': batch[2],
                #         'labels':         batch[3]}

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

    # # Final evaluation
    # _, eval_accuracy = evaluate(args, model, test_dataloader)
    # eps = privacy_engine.get_epsilon(args.delta)
    # print(
    #     f"Final evaluation | " f"Eval accuracy: {eval_accuracy:.3f} | " f"ɛ: {eps:.2f}"
    # )


def get_model():
    model_name = "bert-base-cased"
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=2,
    )
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        config=config,
    )
    trainable_layers = [
        model.bert.encoder.layer[-1],
        model.bert.pooler,
        model.classifier,
    ]
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    print(f"Total parameters count: {total_params}")  # ~108M
    print(f"Trainable parameters count: {trainable_params}")  # ~7M
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Opacus IMDB trained by Transformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset file management arguments
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Where IMDB is/will be stored"
    )
    # Training arguments
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        metavar="B",
        help="input batch size for test",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
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
        default=4e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        metavar="SL",
        help="Longer sequences will be cut to this length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    # parser.add_argument(
    #     "--save-model",
    #     action="store_true",
    #     default=False,
    #     help="Save the trained model",
    # )
    # parser.add_argument(
    #     "--disable-dp",
    #     action="store_true",
    #     default=False,
    #     help="Disable privacy training and just train with vanilla optimizer",
    # )
    # parser.add_argument(
    #     "--secure-rng",
    #     action="store_true",
    #     default=False,
    #     help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    # )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )

    args = parser.parse_args()

    raw_dataset = load_dataset("imdb", cache_dir=args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=True,
            max_length=args.max_sequence_length,
        ),
        batched=True,
    )
    # import pdb;  pdb.set_trace()
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    # Move the model to appropriate device
    model = get_model()
    model = model.to(args.device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Start training
    train(args, model, optimizer, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
