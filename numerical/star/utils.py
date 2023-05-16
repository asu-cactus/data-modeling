import pandas as pd
import numpy as np

from random import sample
from pathlib import Path

# names = {
#     'charge': np.int32,
#     'clus': np.int32,
#     'dst': np.int32,
#     'hist': np.int32,
#     'enumber': np.int32,
#     'etime': np.float64,
#     'rnumber': np.int32,
#     'nlb': np.int32,
#     'qxb': np.float64,
#     'tracks': np.int32,
#     'vertex': np.float64,
#     'zdc': np.int32,
# }

names = {
    'charge': np.int32,
    'clus': np.int32,
    'nlb': np.int32,
    'qxb': np.float64,
    'tracks': np.int32,
    'vertex': np.float64,   
    'zdc': np.int32,
}

patterns = {
    'charge': '{:03d}',
    'clus': '{:04d}',
    'dst': '{:06d}',
    'hist': '{:06d}',
    'enumber': '{:05d}',
    'etime': '{:e}',
    'rnumber': '{:07d}',
    'nlb': '{:04d}',
    'qxb': '{:e}',
    'tracks': '{:04d}',
    'vertex': '{:e}',
    'zdc': '{:03d}',
}

col_names = {
    'charge': np.int32,
    'clus': np.int32,
    'nlb': np.int32,
    'qxb': np.float64, 
    'tracks': np.int32,
    'vertex': np.float64,
    'zdc': np.int32,
}

class DataProcessor:
    def __init__(self):
        self.data = self.load_data()
        self.data_to_list_of_dict()
        
    def load_data(self):
        pathname = "data/star2000_sample.txt"
        filepath = Path(pathname)
        if filepath.is_file():
            with open(filepath, 'r') as f:
                data = [line.strip() for line in f.readlines()]
            return data
        data = pd.read_csv(
            'data/star2000.csv.gz', 
            header=None, 
            usecols=[0,1,7,8,9,10,11],
            names=list(names), 
            dtype=names)
        return self._process_all_rows(data)

    def _num2str(self, num, dtype, pattern):
        if dtype == np.int32:
            return pattern.format(int(num))
        if dtype == np.float64:
            return pattern.format(num)
        raise ValueError(f'No dtype as {dtype}')

    def _row_to_string(self, idx, row):
        features = [
            f"{name}:{self._num2str(row[name], dtype, patterns[name])}" for name, dtype in col_names.items()]
        string = ','.join(features)
        return f'{idx:07}${string}'

    def _process_all_rows(self, data):
        lines = []
        with open('data/star2000.txt',  'w') as f:
            for idx, row in data.iterrows():
                string = self._row_to_string(idx, row)
                f.write(f'{string}\n')
                lines.append(string)
        return lines

    def sample(self, size):
        return sample(self.data, size)
    
    def data_to_list_of_dict(self):
        def line_to_dict(line, delimiter='$'):
            instruction, output = line.split(delimiter, maxsplit=1)
            return {'instruction': f'{instruction}{delimiter}', 'output': output}
        self.data = [line_to_dict(line) for line in self.data]
            
if __name__ == '__main__':
    processor = DataProcessor()
    print(processor.sample_data(2))
    import pdb; pdb.set_trace()
