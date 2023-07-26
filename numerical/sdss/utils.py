import numpy as np
import bz2
import multiprocessing as mp
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Parameters that don't change
DATA_DIR = "data"
# DATA_DIR = "/global/cfs/projectdirs/m1248/john/sdss"
OUTPUT_DIR = "outputs"
NROWS = 168646652
# Make directories
Path.mkdir(Path(OUTPUT_DIR), exist_ok=True)

# Parameters that may change
COLS = ["ab_exp_g", "ab_exp_i", "ab_exp_r", "ab_exp_u", "ab_exp_z"]
NDIGITS = 4


# Function that may change
def preprocess_func(ndarray):
    # Convert -9999 to 0
    ndarray = np.where(ndarray == -9999, 0, ndarray)
    # Multiply by 1000 because we want 4 digits
    ndarray = ndarray * 1000
    # Convert to integers (In fact they are already integers)
    return ndarray.astype(np.int32)


def row_to_string(args):
    idx, row = args
    features = [f"{value:04d}" for value in row]
    string = ",".join(features)
    return {"instruction": f"{idx:09d}$", "output": string}


def load_data(return_ndarray=False):
    # Read the data
    # df_data = {}
    data = []
    for col in COLS:
        with bz2.open(f"{DATA_DIR}/{col}.bz2", "rb") as f:
            data_bytes = f.read()
        data.append(preprocess_func(np.frombuffer(data_bytes, dtype=np.float32)))

    # Create a dataframe
    data = np.stack(data, -1)
    assert data.shape == (NROWS, len(COLS))
    if return_ndarray:
        return data

    # Write data to file
    logger.info("Processing dataframe to strings...")
    with mp.Pool(mp.cpu_count() - 2) as pool:
        data = pool.imap(
            row_to_string, [(i, row) for i, row in enumerate(data)], 1000000
        )
        data = [d for d in data]
        logger.info("Done processing dataframe to strings.")

        # Write data to file for inspection
        data_sample = data[0:2000]
        with open("data/ab_exp_sample.txt", "w") as f:
            for sample in data_sample:
                f.write(f"{sample['instruction']}{sample['output']}\n")

        logger.info("Done writing data to file.")
    return data


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
