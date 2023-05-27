import copy
import logging
from dataclasses import dataclass, field
from typing import Sequence
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import transformers
import utils
from torch.utils.data import Dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
MAX_LENGTH = 50
NGPU = 2
NROWS = 2173762

STAR2000_SPECIAL_TOKENS = [f"{name}:" for name in utils.names]

PROMPT_DICT = {
    "prompt_input": "{intruction}{input}",
    "prompt_no_input": "{instruction}",
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models")
    vocab_size: int = field(default=21)
    hidden_size: int = field(default=128)  # was 512
    intermediate_size: int = field(default=256)  # was 1024
    num_hidden_layers: int = field(default=4)  # was 4
    num_attention_heads: int = field(default=2)  # was 4
    hidden_act: str = field(default="silu")
    max_position_embeddings: int = field(default=MAX_LENGTH)
    initializer_range: float = field(default=0.02)
    rms_norm_eps: float = field(default=1e-06)
    use_cache: bool = field(default=True)
    pad_token_id: int = field(default=19)
    eos_token_id: int = field(default=20)
    tie_word_embeddings: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default="cache")
    model_max_length: int = field(default=MAX_LENGTH)
    output_dir: str = field(default="outputs")
    dataloader_num_workers: int = field(default=16)
    disable_tqdm: bool = field(default=True)
    # # Optimization
    # optim: str = field(default="adamw_torch")
    # learning_rate: float = field(default=2e-4)
    # lr_scheduler_type: str = field(default="linear")
    # Batch size and epochs
    per_device_train_batch_size: int = field(default=256)
    num_train_epochs: float = field(default=10000.0)
    # Logging and saving
    logging_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=10)


def get_optimizer(model, args: TrainingArguments, lr: float = 2e-4):
    optim = transformers.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    num_training_steps = (
        NROWS // (args.per_device_train_batch_size * 2) * NGPU * args.num_train_epochs
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    return (optim, scheduler)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        list_data_dict = utils.DataProcessor().data

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


# def train_tokenizer(
#     model_args: ModelArguments,
# ):
#     # TODO: bypass tokenizer training if it is already done
#     save_dir = model_args.model_name_or_path
#     tokenizer = Tokenizer(Unigram())
#     trainer = UnigramTrainer(
#         vocab_size=model_args.vocab_size,
#         special_tokens=STAR2000_SPECIAL_TOKENS,
#     )
#     files = ['data/star2000.txt']
#     tokenizer.train(files, trainer)
#     tokenizer.save(f'{save_dir}/tokenizer.json')


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_config(
        transformers.LlamaConfig(
            vocab_size=model_args.vocab_size,
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            hidden_act=model_args.hidden_act,
            max_position_embeddings=model_args.max_position_embeddings,
            initializer_range=model_args.initializer_range,
            rms_norm_eps=model_args.rms_norm_eps,
            use_cache=model_args.use_cache,
            pad_token_id=model_args.pad_token_id,
            # bos_token_id=model_args.bos_token_id,
            eos_token_id=model_args.eos_token_id,
            tie_word_embeddings=model_args.tie_word_embeddings,
        )
    )

    tokenizer = transformers.PreTrainedTokenizerFast(
        model_max_length=training_args.model_max_length,
        padding_side="right",
        tokenizer_file="models/star2000_tokenizer.json",
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    utils.estimate_model_size(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=get_optimizer(model, training_args),
        **data_module,
    )
    trainer.train()
    trainer.save_state()


def continue_train(checkpoint: str):
    # TODO: Support dpsgd
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"outputs/{checkpoint}/")
    model = transformers.AutoModelForCausalLM.from_pretrained(f"outputs/{checkpoint}/")
    data_module = make_supervised_data_module(tokenizer=tokenizer)
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    _, training_args = parser.parse_args_into_dataclasses()
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=get_optimizer(model, training_args),
        **data_module,
    )
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
