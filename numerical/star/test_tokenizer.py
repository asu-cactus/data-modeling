from transformers import PreTrainedTokenizerFast

tokenizer_path = 'models/star2000_tokenizer.json'
tokenizer = PreTrainedTokenizerFast(
    padding_side="right", 
    tokenizer_file=tokenizer_path
)

max_len = 0
with open('data/star2000.txt', 'r') as f:
    for line in f:
        tokens = tokenizer.tokenize(line.strip())
        max_len = max(max_len, len(tokens))
        
print(f'Max length: {max_len}') # Max length: 98

        # try:
        #     print(tokenizer.tokenize(line.strip()))
        # except Exception as e:
        #     print(f'Exception: {e}')
        #     print(f'line: {line}')
        #     raise ValueError

