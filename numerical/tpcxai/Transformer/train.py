import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_file = "data/tpcxai_input_sci.txt"
output_dir = "outputs_sci"

from tokenizer_util import create_tokenizer

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Config,
    AutoTokenizer,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TRAIN")


def load_data(tokenizer):
    def preprocess_function(examples):
        nonlocal tokenizer
        inputs_outputs = [example.split("$") for example in examples["text"]]
        inputs, targets = zip(*inputs_outputs)
        # inputs = [f"{input_output[0]}$" for input_output in inputs_outputs]
        # targets = [f"{input_output[1]}<s>" for input_output in inputs_outputs]
        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return model_inputs

    logger.info(f"Loading data from {data_file}")
    dataset = load_dataset("text", data_files={"train": data_file}, split="train")
    iterable_dataset = dataset.to_iterable_dataset()

    updated_dataset = iterable_dataset.map(
        preprocess_function, batched=True, remove_columns=["text"]
    )
    # dataset_iterator = iter(updated_dataset)
    # print(next(dataset_iterator))
    # print(next(dataset_iterator))
    # print(next(dataset_iterator))
    return updated_dataset


def create_model(vocab_size=22):
    model = AutoModelForSeq2SeqLM.from_config(
        T5Config(
            vocab_size=vocab_size,
            d_model=512,
            d_kv=64,
            d_ff=2048,
            num_layers=3,
            num_decoder_layers=6,
            num_heads=2,
            dropout_rate=0,
            classifier_dropout=0,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="relu",  # other option is "gated-gelu"
            use_cache=True,
            decoder_start_token_id=0,
            pad_token_id=1,
            eos_token_id=2,
        )
    )
    size_in_mb = model.get_memory_footprint() / 1024**2
    logger.info(f"Model size is {size_in_mb:.2f}MB")
    logger.info(model.config)
    return model


def train():
    tokenizer = create_tokenizer()
    model = create_model(vocab_size=len(tokenizer))
    dataset = load_data(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Total epoch = per_device_train_batch_size * max_steps * gradient_accumulation_steps / len(dataset)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-3,
        per_device_train_batch_size=512,
        weight_decay=0.01,
        save_total_limit=5,
        max_steps=25_000_000,
        save_steps=1000,
        predict_with_generate=True,
        gradient_accumulation_steps=8,
        fp16=True,
        optim="adafactor",
        push_to_hub=False,
        disable_tqdm=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


def predict():
    checkpoint = "outputs/checkpoint-3000"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    inputs = tokenizer(
        [f"{i:08}<s>" for i in range(100)], return_tensors="pt", padding=True
    )

    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output,
        max_length=128,
    )
    for decoded_text in tokenizer.batch_decode(
        output_sequences,
        skip_special_tokens=True,
    ):
        logger.info("".join(decoded_text.split()))


if __name__ == "__main__":
    # tokenizer = create_tokenizer()
    # load_data(tokenizer)
    train()
    # predict()
    # create_model()
