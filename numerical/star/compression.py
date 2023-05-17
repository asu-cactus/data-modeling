import zstd
import sys
import pandas as pd
from utils import names


def compress_df(df, filename="star2000"):
    # TODO: Make sure if index is included
    compressed = zstd.compress(df.to_records(index=True).tobytes())
    size_in_mb = sys.getsizeof(compressed) / 1024**2
    print(f"Compressed size: {size_in_mb:.2f}MB")
    with open(f"data/{filename}.zstd", "wb") as f:
        f.write(compressed)


if __name__ == "__main__":
    df = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=[2, 3, 4, 5, 6],
        names=list(names),
        dtype=names,
    )
    size_in_mb = sys.getsizeof(df) / 1024**2
    print(f"Before compression size: {size_in_mb:.2f}MB")
    compress_df(df)
