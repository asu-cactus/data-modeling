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
NROWS = 1686466
# NROWS = 168646652
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
        # For now, only read the first NROWS
        data.append(
            preprocess_func(np.frombuffer(data_bytes, dtype=np.float32)[:NROWS])
        )

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
        with open("data/ab_exp.txt", "w") as f:
            for sample in data:
                f.write(f"{sample['instruction']}{sample['output']}\n")

        logger.info("Done writing data to file.")
    return data


def estimate_model_size(model):
    """Estimate Pytorch model size in MB"""
    size_in_mb = model.get_memory_footprint() / 1024**2
    print(f"Model size is {size_in_mb:.2f}MB")
    return size_in_mb


if __name__ == "__main__":
    import random

    data = load_data()
    data_sample = random.sample(data, 1000)

    with open("data/ab_exp.txt", "w") as f:
        f.write(f"{data_sample['instruction']}{data_sample['output']}\n")
