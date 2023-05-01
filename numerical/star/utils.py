import pandas as pd
import numpy as np

names = {
    'charge': np.int32,
    'clus': np.int32,
    'dst': np.int32,
    'hist': np.int32,
    'enumber': np.int32,
    'etime': np.float64,
    'rnumber': np.int32,
    'nlb': np.int32,
    'qxb': np.float64,
    'tracks': np.int32,
    'vertex': np.float64,
    'zdc': np.int32,
}
data = pd.read_csv('star2000.csv', header=None, names=list(names), dtype=names)


class DataProcessor:
    def __init__(self, data, batch_size, use_cols=None):
        if use_cols is None:
            self.col_names = names
        else:
            self.col_names = {col: names[col] for col in use_cols}
        self.batch_size = batch_size
        self.data = data

    def _num2str(self, num, dtype):
        if dtype == np.int32:
            return f'{int(num):d}'
        if dtype == np.float64:
            return f'{num:e}'
        raise ValueError(f'No dtype as {dtype}')

    def _row_to_string(self, idx, row):
        features = [
            f"{col}:{self._num2str(row[col], dtype)}" for col, dtype in self.col_names.items()]
        string = ','.join(features)
        return f'{idx:07}${string}'

    def process_all_rows(self, data):
        with open('star2000.txt',  'w') as f:
            for idx, row in self.data.iterrows():
                string = self._row_to_string(idx, row)
                f.write(f'{string}\n')

    def sample_data(self):
        samples = self.data.sample(self.batch_size, axis=0)
        results = [self._row_to_string(idx, row)
                   for idx, row in samples.iterrows()]
        return results
