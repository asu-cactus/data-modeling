import transformers
import torch
import numpy as np

from train import MAX_LENGTH
from utils import names, COLS, NROWS, DATA_DIR, OUTPUT_DIR, load_data

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CHECKPOINT = "checkpoint-110396"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    prompts = ["000000000$", "000000001$", "000000002$", "000000003$"]
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(prompts, return_tensors="pt").input_ids

    outputs = model.generate(inputs, max_length=MAX_LENGTH, min_length=MAX_LENGTH)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for pred in predictions:
        print(pred)


def parse_pred_lines(lines: list[str]):
    outputs = np.empty((len(lines), len(COLS)), dtype=np.int32)
    for i, line in enumerate(lines):
        num_strs = line.split("$")[1].split(",")
        nums = [int(num_str) for num_str in num_strs]
        if len(nums) != len(COLS):
            nums += [0] * (len(COLS) - len(nums))
        outputs[i] = nums
    return outputs


def predict(batch_size=256):
    prompts = [f"{i:09}$" for i in range(NROWS)]
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    predictions = np.empty((NROWS, len(COLS)), dtype=np.int32)

    for start_idx in range(0, NROWS, batch_size):
        batch = prompts[start_idx : start_idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(inputs, max_length=MAX_LENGTH, min_length=MAX_LENGTH)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = parse_pred_lines(outputs)
        predictions[start_idx : start_idx + len(batch)] = outputs
    # Save predictions as numpy array
    with open(f"{DATA_DIR}/{CHECKPOINT}.npy", "wb") as f:
        np.save(f, predictions)
    return predictions


# def parse_pred_line(line: str):
#     d = {}
#     for name in names:
#         segs = line.split(f"{name}:", maxsplit=1)
#         d[name] = segs[1].split(",", maxsplit=1)[0] if len(segs) > 1 else ""
#     return d


def compute_accuracy(references: np.ndarray, predictions: np.ndarray):
    diff = references - predictions
    n_correct = np.sum(diff == 0, axis=0)
    accuracy = {name: n / NROWS for name, n in zip(names, n_correct)}
    accuracy["all"] = sum(list(accuracy.values())) / len(accuracy)
    return accuracy


# def compute_error_mean(references: pd.DataFrame, predictions: list[str]):
#     # Compute the mean of each column in references, and use it as the default value
#     avgs = references.mean(axis=0)

#     # Compute the average error for each column
#     error_sum = defaultdict(float)
#     for ref, pred in zip(references.values, predictions):
#         pred = parse_pred_line(pred)
#         for i, name in enumerate(names):
#             try:
#                 pred_int = int(pred[name])
#             except ValueError:
#                 pred_int = avgs[name]
#             error_sum[name] += abs(ref[i] - pred_int)
#     error_mean = {name: n / NROWS for name, n in error_sum.items()}
#     error_mean["all"] = sum(list(error_mean.values())) / len(error_mean)
#     return error_mean


def eval_accuracy(predictions=None):
    if predictions is None:
        predictions = np.load(f"{DATA_DIR}/{CHECKPOINT}.npy")
    references = load_data(return_ndarray=True)
    assert predictions.shape == references.shape

    # Eval using accuracy
    accuracy = compute_accuracy(references, predictions)
    print(f"Accuracy:\n{accuracy}")

    # Save inaccurate predictions
    inaccurate_preds = np.where(references != predictions, references, 0)
    with open(f"{OUTPUT_DIR}/{CHECKPOINT}-aux.npy", "wb") as f:
        np.save(f, inaccurate_preds)
    return accuracy


# def eval_avg_error(predictions=None):
#     if predictions is None:
#         predictions = load_lines(f"data/{CHECKPOINT}.txt")
#     references = pd.read_csv(
#         "data/star2000.csv.gz",
#         header=None,
#         usecols=USECOLS,
#         names=list(names),
#         dtype=names,
#     )
#     # Hardcoded: convert all to integer!
#     references = references.astype(int)
#     assert len(predictions) == len(references)

#     # Eval using avg error
#     error_mean = compute_error_mean(references, predictions)
#     print(f"Average absolute error:\n{error_mean}")
#     return error_mean


if __name__ == "__main__":
    predictions = predict()
    # eval_avg_error(predictions)
    eval_accuracy(predictions)

    # eval_accuracy()
    # eval_avg_error()
