import pickle
import numpy as np
from typing import TypeAlias
import random
from sklearn.preprocessing import StandardScaler
from pathlib import Path

Embedding: TypeAlias = np.array


def get_embeddings(size: int) -> list[Embedding]:
    pwd = Path(__file__).parent.resolve()
    with open(pwd / 'data' / '10K_encodings.pkl', 'rb') as f:
        return pickle.load(f)[:size]


def generate_labels(
    size: int,
    label_cardinality: int,
) -> list[list[int]]:
    max_ = 20
    assert label_cardinality >= max_
    no_of_dates = [random.randint(1, max_) for i in range(size)]
    y = np.zeros((size, label_cardinality), dtype=int)
    for i in range(size):
        dates = np.random.choice(
            label_cardinality, no_of_dates[i], replace=False)
        y[i, dates] = 1
    return y


# def generate_labels(
#     size: int,
#     label_cardinality: int,
#     mean: float = 5.0,
#     std: float = 1.0
# ) -> list[list[int]]:
#     pass


def get_data(size: int, label_cardinality: int):
    embeddings = get_embeddings(size)
    labels = generate_labels(size, label_cardinality)
    return (embeddings, labels)


def standardize(embeddings):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings)
    return X_train
