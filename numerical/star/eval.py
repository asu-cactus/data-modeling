import transformers
from train import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN

def get_tokenizer():
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("outputs/checkpoint-5145849/")
    # tokenizer = transformers.PreTrainedTokenizerFast(
    #     model_max_length=100,
    #     padding_side="right", 
    #     tokenizer_file='models/star2000_tokenizer.json'
    # )
    
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        
    # tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def get_model():
    model = transformers.AutoModelForCausalLM.from_pretrained("outputs/checkpoint-5145849/")
    return model
    
def eval():
    prompts = ["0000000$", "0000001$", "0000002$", "0000003$"]
    
    # generator = transformers.pipeline(
    #     "text-generation", 
    #     model='outputs/checkpoint-5145849/',
    #     # tokenizer=get_tokenizer(),
    #     max_new_tokens=100,
    # )

    # print(generator(prompts, clean_up_tokenization_spaces=True, return_tensors=True))
    
    tokenizer = get_tokenizer()
    model = get_model()

    inputs = tokenizer(prompts, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    for result in results:
        print(result)
    
if __name__ == '__main__':
    eval()