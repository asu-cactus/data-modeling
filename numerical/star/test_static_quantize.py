from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
    AutoCalibrationConfig,
)
import pdb

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model)
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)


def preprocess_fn(ex, tokenizer):
    return tokenizer(ex["sentence"])


calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=50,
    dataset_split="train",
)
pdb.set_trace()

calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

inputs = tokenizer("$0001111$", return_tensors="pt")
gen_tokens = onnx_model.generate(**inputs)
print(tokenizer.batch_decode(gen_tokens))


ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    operators_to_quantize=qconfig.operators_to_quantize,
)

model_quantized_path = quantizer.quantize(
    save_dir="checkpoints/quantized/",
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
