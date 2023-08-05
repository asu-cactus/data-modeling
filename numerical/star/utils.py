import pandas as pd
import numpy as np
import logging
from random import sample
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# patterns = {
#     "charge": "{:03d}",
#     "clus": "{:04d}",
#     "dst": "{:06d}",
#     "hist": "{:06d}",
#     "enumber": "{:05d}",
#     "etime": "{:e}",
#     "rnumber": "{:07d}",
#     "nlb": "{:04d}",
#     "qxb": "{:e}",
#     "tracks": "{:04d}",
#     "vertex": "{:e}",
#     "zdc": "{:03d}",
# }

# names = {
#     'charge': np.int32,
#     'clus': np.int32,
#     'dst': np.int32,
#     'hist': np.int32,
#     'enumber': np.int32,
#     'etime': np.float32,
#     'rnumber': np.int32,
#     'nlb': np.int32,
#     'qxb': np.float32,
#     'tracks': np.int32,
#     'vertex': np.float32,
#     'zdc': np.int32,
# }

USECOLS = [2, 3, 4, 5, 6]
names = {
    "dst": np.int32,
    "hist": np.int32,
    "enumber": np.int32,
    "etime": np.float64,
    "rnumber": np.int32,
}


class DataProcessor:
    def __init__(self):
        self.data = self.load_data()
        self.data_to_list_of_dict()

    def load_data(self):
        target_file = Path("data/star2000.txt")
        if target_file.is_file():
            with open(target_file, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                return lines

        data = pd.read_csv(
            "data/star2000.csv.gz",
            header=None,
            usecols=USECOLS,
            names=list(names),
            dtype=names,
        )
        lines = []
        with open(target_file, "w") as f:
            for idx, row in data.iterrows():
                string = self._row_to_string(idx, row)
                f.write(f"{string}\n")
                lines.append(string)
        return lines

    # def _num2str(self, num, dtype, pattern):
    #     if dtype == np.int32:
    #         return pattern.format(int(num))
    #     if dtype == np.float64:
    #         return pattern.format(num)
    #     raise ValueError(f"No dtype as {dtype}")

    # def _row_to_string(self, idx, row):
    #     features = [
    #         f"{name}:{self._num2str(row[name], dtype, patterns[name])}"
    #         for name, dtype in names.items()
    #     ]
    #     string = ",".join(features)
    #     return f"{idx:07}${string}"

    def _num2str(self, num, dtype):
        if dtype == np.int32:
            return f"{int(num):d}"
        if dtype == np.float64:
            return int(float(num))
            # return f"{num:e}"
        raise ValueError(f"No dtype as {dtype}")

    def _row_to_string(self, idx, row):
        features = [
            f"{col}:{self._num2str(row[col], dtype)}" for col, dtype in names.items()
        ]
        string = ",".join(features)
        return f"{idx:07}${string}"

    def data_to_list_of_dict(self):
        def line_to_dict(line, delimiter="$"):
            instruction, output = line.split(delimiter, maxsplit=1)
            return {"instruction": f"{instruction}{delimiter}", "output": output}

        self.data = [line_to_dict(line) for line in self.data]


def sample_data(processor, size):
    return sample(processor.data, size)


def estimate_model_size(model):
    """Estimate Pytorch model size in MB"""
    size_in_mb = model.get_memory_footprint() / 1024**2
    logger.info(f"Model size is {size_in_mb:.2f}MB")
    return size_in_mb


if __name__ == "__main__":
    processor = DataProcessor()
    print(sample_data(processor, 2))
