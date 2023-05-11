import transformers
import torch
import evaluate

from train import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN
from utils import names

NROWS = 2173762
NCOLS = 12
CHECKPOINT = "checkpoint-1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_model_and_tokenizer():
    model = transformers.AutoModelForCausalLM.from_pretrained(f"outputs/{CHECKPOINT}/")
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"outputs/{CHECKPOINT}/")
    return (model, tokenizer)
    
    
def load_lines(path):
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def predict_test():
    prompts = ["0000000$", "0000001$", "0000002$", "0000003$"]
    model, tokenizer = get_model_and_tokenizer()
    
    inputs = tokenizer(prompts, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=110, do_sample=False)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    for pred in predictions:
        print(pred)


def predict(batch_size=256):
    prompts = [f'{i:07}$' for i in range(NROWS)]
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    predictions = []
    with open(f'data/{CHECKPOINT}.txt', 'w') as f:
        for start_idx in range(0, NROWS, batch_size):
            batch = prompts[start_idx: start_idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(inputs, max_new_tokens=110, do_sample=False)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outputs = [text.replace(' ', '').replace(DEFAULT_EOS_TOKEN, '').replace(DEFAULT_PAD_TOKEN, '') for text in outputs]
            predictions.extend(outputs)
            f.writelines([f'{line}\n' for line in outputs])
    return predictions

def parse_line(line):
    return {name: line.split(f'{name}:')[1].split(',', maxsplit=1)[0] for name in names}
    
    
def compute_accuracy(references, predictions):
    n_correct =0
    for ref, pred in zip(references, predictions):
        ref = parse_line(ref)
        pred = parse_line(pred)
        for name in names:
            n_correct += (ref[name] == pred[name])
    return n_correct / (NROWS * NCOLS)
    
def eval(predictions=None):
    if predictions is None:
        predictions= load_lines(f'data/{CHECKPOINT}.txt')
    references = load_lines(f'data/star2000.txt')
    assert len(predictions) == len(references)
    
    # Eval using bleu
    model, tokenizer = get_model_and_tokenizer()
    bleu = evaluate.load("bleu")
    print(bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.tokenize))
    
    # Eval using accuracy
    accuracy = compute_accuracy(references, predictions)
    print(f'Accuracy: {accuracy}')
    
    
if __name__ == '__main__':
    predictions = predict()
    eval(predictions)
    
    # eval()