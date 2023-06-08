import zstd
import sys
import pandas as pd
import numpy as np
import zfpy

names = {
    "dst": np.int32,
    "hist": np.int32,
    "enumber": np.int32,
    "etime": np.float64,
    "rnumber": np.int32,
}


def abs_error(array1, array2):
    abs_error = np.mean(np.abs(array1 - array2), axis=0)
    print(abs_error)


def zstd_compress(df, original_size, save=False):
    compressed = zstd.compress(df.to_records(index=False).tobytes())
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(
        f"Compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
    )
    if save:
        with open(f"data/star2000.zstd", "wb") as f:
            f.write(compressed)
    return size_in_mb


def zfpy_compress(df, original_size, save=False):
    ndarray = df.to_numpy()
    # Reverse mode
    compressed = zfpy.compress_numpy(ndarray)
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(
        f"Reverse mode compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
    )

    # Fixed rate mode
    compressed = zfpy.compress_numpy(ndarray, rate=8)
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(
        f"Fixed rate mode compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
    )
    abs_error(ndarray, zfpy.decompress_numpy(compressed))

    # Fixed precision mode
    compressed = zfpy.compress_numpy(ndarray, precision=32)
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(
        f"Fixed precision mode compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
    )
    abs_error(ndarray, zfpy.decompress_numpy(compressed))

    # Fixed accuracy mode
    df = df.astype(np.float32)
    compressed = zfpy.compress_numpy(ndarray, tolerance=1e16)
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(
        f"Accuracy mode compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
    )
    abs_error(ndarray, zfpy.decompress_numpy(compressed))

    if save:
        with open(f"data/star2000.zfp", "wb") as f:
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
        comp_size_in_mb = zstd_compress(df)
        print(f"Compression ratio: {size_in_mb / comp_size_in_mb:.3f}\n")


def test_compression():
    df = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=[2, 3, 4, 5, 6],
        names=list(names),
        dtype=names,
    )

    # Convert data frame to int32 and add index column
    df = df.astype(np.int32)
    df["index"] = df.index

    # Measure size of original data frame and compressed data frame
    size_in_mb = sys.getsizeof(df) / 1024**2
    print(f"Before compression size: {size_in_mb:.2f}MB")
    zstd_compress(df, original_size=size_in_mb)
    zfpy_compress(df, original_size=size_in_mb)


if __name__ == "__main__":
    test_compression()
