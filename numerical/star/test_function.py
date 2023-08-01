import zstd
import sys
import gzip


def get_size_in_mb(object):
    return sys.getsizeof(object) / 1024**2


def compress_binary_file(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
    original_size = get_size_in_mb(content)
    print(f"Original size: {original_size} MB")

    compressed_content = zstd.compress(content, 22)
    compressed_size = get_size_in_mb(compressed_content)
    print(f"Zstd compressed size: {compressed_size} MB")


# compress_binary_file(f"outputs/checkpoint-2052941/pytorch_model.bin")
compress_binary_file(f"outputs_pruning/checkpoint-16984000-quantized/pytorch_model.bin")
