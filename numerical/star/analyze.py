from eval import load_lines
from utils import names

from collections import defaultdict

def parse_pred_line(line: str):
    d = {}
    for name in names:
        segs = line.split(f'{name}:', maxsplit=1)
        d[name] = segs[1].split(',', maxsplit=1)[0] if len(segs) > 1 else ''
    return d

def extract_incorrect_outputs(checkpoint: str):
    """ Compare star2000.txt and checkout-x.txt """
    predictions = load_lines(f'data/{checkpoint}.txt')
    references = load_lines(f'data/star2000.txt')
    assert len(predictions) == len(references)
    
    mismatch = defaultdict(list)
    for ref, pred in zip(references, predictions):
        ref = {name: ref.split(f'{name}:', maxsplit=1)[1].split(
            ',', maxsplit=1)[0]for name in names}
        pred = parse_pred_line(pred)
        for name in names:
            if ref[name] != pred[name]:
                mismatch[name].append((ref[name], pred[name]))
    for name in names.keys():
        with open(f'{name}.out', 'w') as f:
            f.writelines([f'{ref},{pred}\n' for ref, pred in mismatch[name]])
    
def _extract_correct_output_rows(checkpoint: str):
    """ Compare star2000.txt and checkout-x.txt """
    predictions = load_lines(f'data/{checkpoint}.txt')
    references = load_lines(f'data/star2000.txt')
    assert len(predictions) == len(references)
    
    results = defaultdict(list)
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        ref = {name: ref.split(f'{name}:', maxsplit=1)[1].split(
            ',', maxsplit=1)[0]for name in names}
        pred = parse_pred_line(pred)
        for name in names.keys():
            if ref[name] == pred[name]:
                results[name].append(i)
    return results
                
def get_overlap(cpt1: str, cpt2: str):
    d1 = _extract_correct_output_rows(cpt1)
    d2 = _extract_correct_output_rows(cpt2)

    for name in names.keys():
        overlap =  set(d1[name]).intersection(d2[name])
        print(f"""{name}:
              checkpoint1 corrects: {len(d1[name])}
              checkpoint2 corrects: {len(d2[name])}
              overlap: {len(overlap)}""")
        
get_overlap('checkpoint-33968', 'checkpoint-46706')