import transformers
import torch
import numpy as np
from scipy import sparse
from transformers import BitsAndBytesConfig

from train_v2 import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, MAX_LENGTH, NROWS
from utils_v2 import estimate_model_size, COLS

from collections import defaultdict
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

OUTPUT_DIR = "outputs_v2"
CHECKPOINT = "checkpoint-1658063"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# quantization = "bitsandbytes8bit"
quantization = None


def get_model_and_tokenizer():
    if quantization == None:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            f"{OUTPUT_DIR}/{CHECKPOINT}/"
        )
    elif quantization == "bitsandbytes4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            f"{OUTPUT_DIR}/{CHECKPOINT}/", quantization_config=quantization_config
        )
    elif quantization == "bitsandbytes8bit":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            f"{OUTPUT_DIR}/{CHECKPOINT}/", device_map="auto", load_in_8bit=True
        )
    estimate_model_size(model)

    # model.save_pretrained(f"{OUTPUT_DIR}/{CHECKPOINT}-quantized/", from_pt=True)
    # exit()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        f"{OUTPUT_DIR}/{CHECKPOINT}/"
    )
    return (model, tokenizer)


def load_lines(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def predict_test(batch_size=1000):
    prompts = [f"{i:07}$" for i in range(10000)]
    model, tokenizer = get_model_and_tokenizer()
    if quantization is None:
        model = model.to(device)

    predictions = []

    for start_idx in range(0, NROWS, batch_size):
        batch = prompts[start_idx : start_idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            inputs,
            max_length=MAX_LENGTH,
            min_length=MAX_LENGTH,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        outputs = [
            text.replace(" ", "")
            .replace(DEFAULT_EOS_TOKEN, "")
            .replace(DEFAULT_PAD_TOKEN, "")
            for text in outputs
        ]

    return predictions


def predict(batch_size=256):
    prompts = [f"{i:07}$" for i in range(NROWS)]
    model, tokenizer = get_model_and_tokenizer()
    if quantization is None:
        model = model.to(device)

    predictions = []
    with open(f"data/{CHECKPOINT}.txt", "w") as f:
        for start_idx in range(0, NROWS, batch_size):
            batch = prompts[start_idx : start_idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                inputs,
                max_length=MAX_LENGTH,
                min_length=MAX_LENGTH,
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


def compute_accuracy(references, predictions, compute_error=False):
    n_correct = defaultdict(int)
    error_sum = defaultdict(int)
    aux_structure = np.zeros((len(references), len(COLS)), dtype=np.int32)
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        refs = ref.split("$")[1].split(",")
        preds = pred.split("$")[1].split(",")
        preds += ["0"] * (len(COLS) - len(preds))
        for j, (ref_val, pred_val, name) in enumerate(zip(refs, preds, COLS)):
            if ref_val == pred_val:
                n_correct[name] += 1
            else:
                aux_structure[i][j] = int(ref_val)

            if compute_error:
                error_sum[name] += abs(int(ref_val) - int(pred_val))
    accuracy = {name: n / NROWS for name, n in n_correct.items()}
    accuracy["all"] = sum(list(accuracy.values())) / len(accuracy)

    logger.info(f"Accuracy:\n{accuracy}")

    if compute_error:
        error_mean = {name: n / NROWS for name, n in error_sum.items()}
        error_mean["all"] = sum(list(error_mean.values())) / len(error_mean)
        logger.info(f"Error mean:\n{error_mean}")

    # Save auxilary structure
    aux_structure_sparse = sparse.csc_matrix(aux_structure)
    sparse.save_npz(f"data/{CHECKPOINT}-aux.npz", aux_structure_sparse)
    aux_size_in_kb = (
        aux_structure_sparse.data.nbytes
        + aux_structure_sparse.indptr.nbytes
        + aux_structure_sparse.indices.nbytes
    ) / 1024
    logger.info(f"{aux_size_in_kb}KB")

    return accuracy


def eval_accuracy(predictions=None):
    if predictions is None:
        predictions = load_lines(f"data/{CHECKPOINT}.txt")
    references = load_lines(f"data/star2000_v2.txt")
    assert len(predictions) == len(references)

    # Eval using accuracy
    accuracy = compute_accuracy(references, predictions)
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
#     logger.info(f"Average absolute error:\n{error_mean}")
#     return error_mean


if __name__ == "__main__":
    predictions = predict()
    eval_accuracy(predictions)
    # eval_avg_error(predictions)

    # eval_accuracy()
    # eval_avg_error()
