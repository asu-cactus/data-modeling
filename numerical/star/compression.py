import zstd
import sys
import pandas as pd
import numpy as np
import zfpy
from third_party.pysz import (
    SZ,
)  # remember to include "third_party" folder in LD_LIBRARY_PATH


def abs_error(array1, array2):
    abs_error = np.mean(np.abs(array1 - array2), axis=0)
    # print(abs_error)
    return abs_error.mean()


def get_size_in_mb(object):
    return sys.getsizeof(object) / 1024**2


def zstd_compress(df):
    for col_name in df.columns:
        column = df[[col_name]].to_records(index=False).tobytes()
        original_size = get_size_in_mb(column)
        compressed = zstd.compress(column)
        compress_size = get_size_in_mb(compressed)
        print(
            f"Column {col_name}: compressed size: {compress_size:.2f}MB, "
            f"compression ratio: {original_size / compress_size:.3f}"
        )


def zfpy_compress(df, original_size, mode):
    ndarray = df.to_numpy()
    # Reverse mode
    if mode == "reverse":
        compressed = zfpy.compress_numpy(ndarray)
        size_in_mb = sys.getsizeof(compressed) / 1024**2
        print(
            f"Reverse mode compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
        )

    # Fixed rate mode
    elif mode == "fixed_rate":
        result = {"rate": [], "compression_ratio": [], "log_abs_error": []}
        for rate in range(2, 16, 2):
            compressed = zfpy.compress_numpy(ndarray, rate=rate)
            size_in_mb = sys.getsizeof(compressed) / 1024**2
            compr_ratio = original_size / size_in_mb
            print(f"Rate: {rate}:")
            print(
                f"Fixed rate mode compressed size: {size_in_mb:.2f}MB, compression ratio: {compr_ratio:.3f}"
            )
            avg_abs_error = abs_error(ndarray, zfpy.decompress_numpy(compressed))
            result["rate"].append(rate)
            result["compression_ratio"].append(compr_ratio)
            result["log_abs_error"].append(np.log(avg_abs_error))
        pd.DataFrame.from_dict(result).to_csv("outputs/fixed_rate.csv", index=False)

    # Fixed precision mode
    elif mode == "fixed_precision":
        result = {"precision": [], "compression_ratio": [], "log_abs_error": []}
        for precision in range(16, 31, 2):
            compressed = zfpy.compress_numpy(ndarray, precision=precision)
            size_in_mb = sys.getsizeof(compressed) / 1024**2
            compr_ratio = original_size / size_in_mb
            print(
                f"Fixed precision mode compressed size: {size_in_mb:.2f}MB, compression ratio: {compr_ratio:.3f}"
            )
            avg_abs_error = abs_error(ndarray, zfpy.decompress_numpy(compressed))
            result["precision"].append(precision)
            result["compression_ratio"].append(compr_ratio)
            result["log_abs_error"].append(np.log(avg_abs_error))
        pd.DataFrame.from_dict(result).to_csv(
            "outputs/fixed_precision.csv", index=False
        )

    # Fixed accuracy mode
    elif mode == "fixed_accuracy":
        df = df.astype(np.float32)
        for tolerance in [1e-2, 1e10]:
            compressed = zfpy.compress_numpy(ndarray, tolerance=tolerance)
            size_in_mb = sys.getsizeof(compressed) / 1024**2
            print(
                f"Accuracy mode compressed size: {size_in_mb:.2f}MB, compression ratio: {original_size / size_in_mb:.3f}"
            )
            abs_error(ndarray, zfpy.decompress_numpy(compressed))


def sz_compress(df, mode):
    # mode_convert = {
    #     "ABS": 0,
    #     "REL": 1,
    #     "ABS_AND_REL": 2,
    #     "ABS_OR_REL": 3,
    #     "PSNR": 4,
    #     "NORM": 5,
    #     "PW_RE": 10,
    # }

    df = df.astype(np.float32)
    # ndarray = df.to_numpy()
    sz = SZ("third_party/libSZ3c.so")

    if mode == "REL":  # Doesn't change
        # result = {"eb_rel": [], "compression_ratio": [], "log_abs_error": []}
        # for eb_rel in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]:
        #     compressed, _ = sz.compress(ndarray, 1, 0, eb_rel, 0)
        #     size_in_mb = sys.getsizeof(compressed) / 1024**2
        #     compr_ratio = original_size / size_in_mb
        #     print(
        #         f"REL mode compressed size: {size_in_mb:.2f}MB, compression ratio: {compr_ratio:.3f}"
        #     )
        #     avg_abs_error = abs_error(
        #         ndarray, sz.decompress(compressed, ndarray.shape, ndarray.dtype)
        #     )
        #     result["eb_rel"].append(eb_rel)
        #     result["compression_ratio"].append(compr_ratio)
        #     result["log_abs_error"].append(np.log(avg_abs_error))
        # pd.DataFrame.from_dict(result).to_csv("outputs/rel.csv", index=False)

        for col_name in df.columns:
            column = df[col_name].to_numpy()
            original_size = get_size_in_mb(df[col_name])
            compressed, _ = sz.compress(column, 1, 0, 1e-4, 0)
            compress_size = get_size_in_mb(compressed)
            avg_abs_error = abs_error(
                column, sz.decompress(compressed, column.shape, column.dtype)
            )
            print(
                f"Column {col_name}: compressed size: {compress_size:.2f}MB, "
                f"compression ratio: {original_size / compress_size:.3f}, "
                f"error: {avg_abs_error:.3f}"
            )


def test_compression():
    names = {
        "dst": np.int32,
        "hist": np.int32,
        "enumber": np.int32,
        "etime": np.float64,
        "rnumber": np.int32,
    }

    df = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=[2, 3, 4, 5, 6],
        names=list(names),
        dtype=names,
    )

    # Convert data frame to int32 and add index column
    df = df.astype(np.int32)
    # df["index"] = df.index

    # Measure size of original data frame and compressed data frame
    size_in_mb = sys.getsizeof(df) / 1024**2
    print(f"Before compression size: {size_in_mb:.2f}MB")
    # zstd_compress(df)
    # zfpy_compress(df, mode="fixed_rate")

    sz_compress(df, mode="REL")


if __name__ == "__main__":
    test_compression()
