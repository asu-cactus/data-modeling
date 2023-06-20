import transformers
import torch
import pandas as pd

from train import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, MAX_LENGTH
from utils import names, USECOLS

from collections import defaultdict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CHECKPOINT = "checkpoint-5099446"
NROWS = 2173762
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    model = transformers.AutoModelForCausalLM.from_pretrained(f"outputs/{CHECKPOINT}/")
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"outputs/{CHECKPOINT}/")
    return (model, tokenizer)


def load_lines(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def predict_test():
    prompts = ["0000000$", "0000001$", "0000002$", "0000003$"]
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(prompts, return_tensors="pt").input_ids

    max_new_tokens = MAX_LENGTH + 10
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    for pred in predictions:
        print(pred)


def predict(batch_size=256):
    prompts = [f"{i:07}$" for i in range(NROWS)]
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    max_new_tokens = MAX_LENGTH + 10
    predictions = []
    with open(f"data/{CHECKPOINT}.txt", "w") as f:
        for start_idx in range(0, NROWS, batch_size):
            batch = prompts[start_idx : start_idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outputs = [
                text.replace(" ", "")
                .replace(DEFAULT_EOS_TOKEN, "")
                .replace(DEFAULT_PAD_TOKEN, "")
                for text in outputs
            ]
            predictions.extend(outputs)
            f.writelines([f"{line}\n" for line in outputs])
    return predictions


def parse_pred_line(line: str):
    d = {}
    for name in names:
        segs = line.split(f"{name}:", maxsplit=1)
        d[name] = segs[1].split(",", maxsplit=1)[0] if len(segs) > 1 else ""
    return d


def compute_accuracy(references, predictions):
    n_correct = defaultdict(int)
    for ref, pred in zip(references, predictions):
        ref = {
            name: ref.split(f"{name}:", maxsplit=1)[1].split(",", maxsplit=1)[0]
            for name in names
        }
        pred = parse_pred_line(pred)
        for name in names:
            n_correct[name] += ref[name] == pred[name]
    accuracy = {name: n / NROWS for name, n in n_correct.items()}
    accuracy["all"] = sum(list(accuracy.values())) / len(accuracy)
    return accuracy


def compute_error_mean(references: pd.DataFrame, predictions: list[str]):
    # Compute the mean of each column in references, and use it as the default value
    avgs = references.mean(axis=0)

    # Compute the average error for each column
    error_sum = defaultdict(float)
    for ref, pred in zip(references.values, predictions):
        pred = parse_pred_line(pred)
        for i, name in enumerate(names):
            try:
                pred_int = int(pred[name])
            except ValueError:
                pred_int = avgs[name]
            error_sum[name] += abs(ref[i] - pred_int)
    error_mean = {name: n / NROWS for name, n in error_sum.items()}
    error_mean["all"] = sum(list(error_mean.values())) / len(error_mean)
    return error_mean


def eval_accuracy(predictions=None):
    if predictions is None:
        predictions = load_lines(f"data/{CHECKPOINT}.txt")
    references = load_lines(f"data/star2000.txt")
    assert len(predictions) == len(references)

    # # Eval using bleu
    # import evaluate
    # model, tokenizer = get_model_and_tokenizer()
    # bleu = evaluate.load("bleu")
    # print(bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.tokenize))

    # Eval using accuracy
    accuracy = compute_accuracy(references, predictions)
    print(f"Accuracy:\n{accuracy}")
    return accuracy


def eval_avg_error(predictions=None):
    if predictions is None:
        predictions = load_lines(f"data/{CHECKPOINT}.txt")
    references = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=USECOLS,
        names=list(names),
        dtype=names,
    )
    # Hardcoded: convert all to integer!
    references = references.astype(int)
    assert len(predictions) == len(references)

    # Eval using avg error
    error_mean = compute_error_mean(references, predictions)
    print(f"Average absolute error:\n{error_mean}")
    return error_mean


if __name__ == "__main__":
    predictions = predict()
    eval_avg_error(predictions)
    eval_accuracy(predictions)

    # eval_accuracy()
    # eval_avg_error()
