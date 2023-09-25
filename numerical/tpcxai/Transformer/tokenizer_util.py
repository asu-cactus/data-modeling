from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Digits

from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def create_tokenizer():
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Digits(individual_digits=True)
    tokenizer.add_special_tokens(["<unk>", "<pad>", "<s>"])
    # Note: "$" is unnecessary because it is part of the encoder, but it may help the decoder to learn the pattern.
    tokenizer.add_tokens(
        [
            # "$",
            ",",
            ".",
            "amount:",
            "sendID:",
            "recID:",
            "tranID:",
            "time:",
            "date:",
            "e",
        ]
        + list("0123456789")
    )
    tokenizer.post_processor = TemplateProcessing(
        single="$A <s>",
        pair="$A $B <s>",
        special_tokens=[("<s>", 2)],
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<s>",
        unk_token="<unk>",
    )
    print(f"Vocab size: {len(fast_tokenizer)}")
    return fast_tokenizer


if __name__ == "__main__":
    tokenizer = create_tokenizer()
    print(tokenizer.vocab_size)
    print(len(tokenizer))
    input_str = "00000000$amount:23947e3,sendID:160344e6,recID:55821816e8,time:92e3,date:4072012e7,tranID:65616255e12"
    print(tokenizer.tokenize(input_str))
    print(
        tokenizer(input_str, return_attention_mask=False, return_token_type_ids=False)
    )
