import pandas as pd
import numpy as np
import pdb


def convert_to_float32(nrows=100_000_000):
    df = pd.read_csv(
        "data/tpcxai_fraud_test.csv",
        header=None,
        usecols=[0, 2, 3, 4, 5, 6],
        dtype=np.float32,
        nrows=nrows,
    )
    df.to_csv("data/tpcxai_fraud_float32.csv", index=False, header=False)
    with open("data/tpcxai_fraud_float32.npy", "wb") as f:
        np.save(f, df.to_numpy())


def to_text_input_format(use_scientific_notation=True):
    """Base on tpcxai_fraud_float32.csv (plain text), convert the float32 data to the format that can be used by the language model."""
    if use_scientific_notation:
        with open("data/tpcxai_input_sci.txt", "w") as output_file:
            with open("data/tpcxai_fraud_float32.csv", "r") as input_file:
                for i, line in enumerate(input_file):
                    nums = line.split(",")
                    base_part = [
                        "".join(num.split(".")).rstrip().rstrip("0") for num in nums
                    ]
                    exp_part = [num.index(".") for num in nums]
                    combined = [
                        f"{base}e{exp}" for base, exp in zip(base_part, exp_part)
                    ]
                    output_file.write(
                        f"{i:08}$amount:{combined[0]},sendID:{combined[1]},recID:{combined[2]},time:{combined[4]},date:{combined[5]},tranID:{combined[3]}\n"
                    )
    else:
        with open("data/tpcxai_input.txt", "w") as output_file:
            with open("data/tpcxai_fraud_float32.csv", "r") as input_file:
                for i, line in enumerate(input_file):
                    nums = line.split(",")
                    output_file.write(
                        f"{i:08}$amount:{nums[0]},senderID:{int(float(nums[1]))},recID:{int(float(nums[2]))},time:{int(float(nums[4]))},date:{int(float(nums[5]))},tranID:{int(float(nums[3]))}\n"
                    )

    # df = pd.read_csv(
    #     "data/tpcxai_fraud_float32.csv", nrows=10, header=None, dtype=np.float32
    # )
    # for i, row in df.iterrows():
    #     # values_start_from_second = [int(num) for num in row[1:]]
    #     print(
    #         f"{i:08}$amount{row[0]:.2f}senderID{int(row[1])}recID{int(row[2])}tranID{int(row[3])}time{int(row[4])}date{int(row[5])}"
    #     )
    #     pdb.set_trace()


if __name__ == "__main__":
    to_text_input_format()
    to_text_input_format(use_scientific_notation=False)
