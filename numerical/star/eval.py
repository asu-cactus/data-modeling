import torch
import pandas as pd
import numpy as np
from scipy import sparse
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

from train import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, MAX_LENGTH
from utils import names, USECOLS, estimate_model_size

# import pdb
import time
from collections import defaultdict
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

OUPUT_DIR = "outputs"
CHECKPOINT = "checkpoint-1919192"
NROWS = 2173762
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# quantization = "inc"
quantization = None


def create_ort_quantized_model():
    from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
    from optimum.onnxruntime.configuration import (
        AutoQuantizationConfig,
        AutoCalibrationConfig,
    )

    onnx_model = ORTModelForCausalLM.from_pretrained(
        f"{OUPUT_DIR}/{CHECKPOINT}/", export=False
    )

    tokenizer = AutoTokenizer.from_pretrained(f"{OUPUT_DIR}/{CHECKPOINT}/")
    quantizer = ORTQuantizer.from_pretrained(onnx_model)
    qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)

    calibration_dataset = Dataset.from_dict(
        {"prompt": [f"{i:07}$" for i in range(100)]}
    ).map(lambda ex: tokenizer(ex["prompt"]), batched=True)

    # pdb.set_trace()
    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    ranges = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
    )
    logger.info(f"Starting quantization...")
    model_quantized_path = quantizer.quantize(
        save_dir=f"{OUPUT_DIR}/quantized/",
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
    )
    logger.info(f"Quantized model saved to {model_quantized_path}")
    return model_quantized_path


def create_inc_quantized_model():
    from neural_compressor.config import (
        PostTrainingQuantConfig,
        AccuracyCriterion,
        TuningCriterion,
    )
    from optimum.intel import INCQuantizer

    model = AutoModelForCausalLM.from_pretrained(f"{OUPUT_DIR}/{CHECKPOINT}/")
    tokenizer = AutoTokenizer.from_pretrained(f"{OUPUT_DIR}/{CHECKPOINT}/")

    # Set up quantization config, TODO: Distributed Acuracy-aware Tuning.
    recipes = {
        "smooth_quant": True,
        "smooth_quant_args": {"alpha": 0.5, "folding": True},
    }
    # Set the accepted accuracy loss to 5%
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.5)
    # Set the maximum number of trials to 10
    tuning_criterion = TuningCriterion(max_trials=100)
    quantization_config = PostTrainingQuantConfig(
        approach="static",
        backend="ipex",
        recipes=recipes,
        accuracy_criterion=accuracy_criterion,
        tuning_criterion=tuning_criterion,
    )

    calibration_dataset = (
        Dataset.from_dict({"prompt": [f"{i:07}$" for i in range(50)]})
        .map(
            lambda ex: tokenizer(ex["prompt"], return_token_type_ids=False),
            batched=True,
        )
        .remove_columns("prompt")
    )
    # import pdb

    # pdb.set_trace()

    quantizer = INCQuantizer.from_pretrained(model)
    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=calibration_dataset,
        save_directory=f"{CHECKPOINT}/inc_quantized/",
    )
    exit()
    return (model, tokenizer)


def get_model_and_tokenizer():
    tokenizer = None
    match quantization:
        case "bitsandbytes4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                f"{OUPUT_DIR}/{CHECKPOINT}/", quantization_config=quantization_config
            )

        case "bitsandbytes8bit":
            model = AutoModelForCausalLM.from_pretrained(
                f"{OUPUT_DIR}/{CHECKPOINT}/", device_map=0, load_in_8bit=True
            )
        case "ort":
            model_quantized_path = create_ort_quantized_model()
            exit()
        case "inc":
            model, tokenizer = create_inc_quantized_model()
        case _:
            logger.info("Loading original model...")
            model = AutoModelForCausalLM.from_pretrained(f"{OUPUT_DIR}/{CHECKPOINT}/")
    estimate_model_size(model)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(f"{OUPUT_DIR}/{CHECKPOINT}/")
    return (model, tokenizer)


def load_lines(path):
    return [line.strip() for line in open(path, "r")]


def predict_test(batch_size=1000):
    test_rows = 10000
    prompts = [f"{i:07}$" for i in range(test_rows)]
    model, tokenizer = get_model_and_tokenizer()
    if quantization is None:
        model = model.to(device)
    max_new_tokens = MAX_LENGTH + 10
    start_time = time.time()
    for start_idx in range(0, test_rows, batch_size):
        batch = prompts[start_idx : start_idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        outputs = [
            text.replace(" ", "")
            .replace(DEFAULT_EOS_TOKEN, "")
            .replace(DEFAULT_PAD_TOKEN, "")
            for text in outputs
        ]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def predict(batch_size=256):
    prompts = [f"{i:07}$" for i in range(NROWS)]
    model, tokenizer = get_model_and_tokenizer()
    if quantization is None:
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
    aux_structure = np.zeros((len(references), len(names)), dtype=np.int32)
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        ref = {
            name: ref.split(f"{name}:", maxsplit=1)[1].split(",", maxsplit=1)[0]
            for name in names
        }
        pred = parse_pred_line(pred)
        for j, name in enumerate(names):
            if ref[name] == pred[name]:
                n_correct[name] += 1
            else:
                aux_structure[i][j] = int(ref[name])
    accuracy = {name: n / NROWS for name, n in n_correct.items()}
    accuracy["all"] = sum(list(accuracy.values())) / len(accuracy)
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
    references = load_lines(f"data/star2000.txt")
    assert len(predictions) == len(references)

    # Eval using accuracy
    accuracy = compute_accuracy(references, predictions)
    logger.info(f"Accuracy:\n{accuracy}")
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
    logger.info(f"Average absolute error:\n{error_mean}")
    return error_mean


if __name__ == "__main__":
    predictions = predict()
    eval_accuracy(predictions)

    # predict_test()
