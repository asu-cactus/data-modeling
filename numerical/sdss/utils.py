import pandas as pd
import numpy as np
import bz2
import multiprocessing as mp
from pathlib import Path
import logging

# Parameters that don't change
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
NROWS = 168646652
# Make directories
Path.mkdir(Path(DATA_DIR), exist_ok=True)
Path.mkdir(Path(OUTPUT_DIR), exist_ok=True)

# Parameters that may change
COLS = ["ab_exp_g", "ab_exp_i", "ab_exp_r", "ab_exp_u", "ab_exp_z"]
NDIGITS = 4


# Function that may change
def preprocess_func(ndarray):
    return np.where(ndarray == -9999, 0, ndarray)


# Function that may change
def row_to_string(row):
    features = [f"int{row[col] * 3}:04" for col in COLS]
    string = ",".join(features)
    return string


def load_data():
    # Read the data
    df_data = {}
    for col in COLS:
        with bz2.open(f"/global/cfs/projectdirs/m1248/john/sdss/{col}.bz2", "rb") as f:
            data = f.read()
        df_data[col] = preprocess_func(np.frombuffer(data, dtype=np.float32))

    # Create a dataframe
    logging.warning("Creating dataframe...")
    df = pd.DataFrame(df_data)

    # Write data to file
    logging.warning("Processing dataframe to strings...")
    pool = mp.Pool()
    row_strings = pool.map(row_to_string, [row for _, row in df.iterrows()])
    del df
    return [
        {"instruction": f"{idx:09}$", "output": row_string}
        for idx, row_string in enumerate(row_strings)
    ]
    # lines = []
    # for idx, row in df.iterrows():
    #     row_string = row_to_string(row)
    #     lines.append({"instruction": f"{idx:09}$", "output": row_string})
    # return lines


def estimate_model_size(model):
    """Estimate Pytorch model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Parameter size is {param_size / 1024**2:.2f}MB")
    print(f"Buffer size is {buffer_size / 1024**2:.2f}MB")
    print(f"Model size is {size_all_mb:.2f}MB")
    return size_all_mb


if __name__ == "__main__":
    import random

    data = load_data()
    data_sample = random.sample(data, 1000)

    with open("data/ab_exp.txt", "w") as f:
        f.write(f"{data_sample['instruction']}{data_sample['output']}\n")
