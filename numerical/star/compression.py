import zstd
import sys
import pandas as pd
import numpy as np

from utils import names


def compress_df(df, save=False):
    compressed = zstd.compress(df.to_records(index=True).tobytes())
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(f"Compressed size: {size_in_mb:.2f}MB")
    if save:
        with open(f"data/star2000.zstd", "wb") as f:
            f.write(compressed)
    return size_in_mb


def missing_one_attr_experiment():
    for i, name in enumerate(names.keys()):
        print(f"Missing attribute: {name}:")
        usenames = {key: value for key, value in names.items() if key != name}
        usecols = [j for j in range(len(names)) if j != i]
        df = pd.read_csv(
            "data/star2000.csv.gz",
            header=None,
            usecols=usecols,
            names=list(usenames),
            dtype=usenames,
        )
        size_in_mb = sys.getsizeof(df) / 1024**2
        print(f"Before compression size: {size_in_mb:.2f}MB")
        comp_size_in_mb = compress_df(df)
        print(f"Compression ratio: {comp_size_in_mb / size_in_mb:.3f}\n")


def test_compression():
    df = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=[2, 3, 4, 5],
        names=list(names),
        dtype=names,
    )
    size_in_mb = sys.getsizeof(df) / 1024**2
    print(f"Before compression size: {size_in_mb:.2f}MB")
    comp_size_in_mb = compress_df(df)
    print(f"Compression ratio: {comp_size_in_mb / size_in_mb:.3f}\n")


if __name__ == "__main__":
    test_compression()
