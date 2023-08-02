import pandas as pd
import numpy as np
import multiprocessing as mp
from random import sample
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLS = ["enumber", "etime", "rnumber", "dst", "hist"]
patterns = [
    "{:05d}",
    "{:08d}",
    "{:07d}",
    "{:06d}",
    "{:06d}",
]

# COLS = ["dst", "hist", "enumber", "etime", "rnumber"]
# patterns = [
#     "{:06d}",
#     "{:06d}",
#     "{:05d}",
#     "{:08d}",
#     "{:07d}",
# ]


def _row_to_dict(row):
    instruction = f"{row[0]:05d},{row[1]:08d}$"
    output = ",".join(
        [pattern.format(value) for value, pattern in zip(row[2:], patterns)]
    )
    return {"instruction": instruction, "output": output}


def load_data(is_shuffle=False):
    USECOLS = [2, 3, 4, 5, 6]
    names = {
        "dst": np.int32,
        "hist": np.int32,
        "enumber": np.int32,
        "etime": np.float64,
        "rnumber": np.int32,
    }
    data = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=USECOLS,
        names=list(names),
        dtype=names,
    )
    if is_shuffle:
        data = data.sample(frac=1, random_state=42)
    # Reorder columns
    data = data[COLS]
    data = data.to_numpy().astype(np.int32)

    with mp.Pool(mp.cpu_count() - 2) as pool:
        data = pool.imap(_row_to_dict, [row for row in data], 100000)
        data = [d for d in data]

    with open("data/star2000.txt", "w") as f:
        for sample in data:
            f.write(f"{sample['instruction']}{sample['output']}\n")
    return data


def sample_data(data, size):
    return sample(data, size)


def estimate_model_size(model):
    """Estimate Pytorch model size in MB"""
    size_in_mb = model.get_memory_footprint() / 1024**2
    logger.info(f"Model size is {size_in_mb:.2f}MB")
    return size_in_mb


if __name__ == "__main__":
    data = load_data()
    print(sample_data(data, 5))
