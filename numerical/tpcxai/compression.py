from third_party.pysz import (
    SZ,
)
import numpy as np


def compute_error(array1, array2, mode="abs", print_errors=False):
    if mode == "abs":
        errors = np.mean(np.abs(array1 - array2), axis=0)
    elif mode == "mse":
        errors = np.mean((array1 - array2) ** 2, axis=0)

    if print_errors:
        print(f"errors: {errors}")
    return errors.mean()


def sz_compress(combine=False):  # REL mode
    sz = SZ("third_party/libSZ3c.so")
    with open("data/tpcxai_fraud_float32.npy", "rb") as f:
        ndarray = np.load(f)

    if combine:
        cmp_data, cmp_ratio, cmp_size = sz.compress(ndarray, 1, 0, 1e-4, 0)
        avg_abs_error = compute_error(
            ndarray,
            sz.decompress(cmp_data, ndarray.shape, ndarray.dtype),
            print_errors=True,
        )
        print(
            f"Compressed size: {cmp_size:.2f}MB, "
            f"compression ratio: {cmp_ratio:.3f}, "
            f"error: {avg_abs_error:.3f}"
        )
    else:
        for col in range(ndarray.shape[1]):
            column = ndarray[:, col].reshape(-1)
            cmp_data, cmp_ratio, cmp_size = sz.compress(column, 1, 0, 1e-4, 0)

            avg_abs_error = compute_error(
                column, sz.decompress(cmp_data, column.shape, column.dtype)
            )
            print(
                f"Column {col}: compressed size: {cmp_size:.2f}MB, "
                f"compression ratio: {cmp_ratio:.3f}, "
                f"error: {avg_abs_error:.6f}"
            )


if __name__ == "__main__":
    sz_compress(combine=False)
