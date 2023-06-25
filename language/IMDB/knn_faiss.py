import faiss
import numpy as np


def load_embeddings(split="train"):
    if split == "train":
        embeddings = np.load("data/train_vectors.npy")
        labels = np.load("data/train_labels.npy")
    else:
        embeddings = np.load("data/test_vectors.npy")
        labels = np.load("data/test_labels.npy")

    return (embeddings, labels)


def add_noise_to_train_embeddings(train_embeddings, epsilon):
    """Add noise to train embeddings,
    according to paper https://dl.acm.org/doi/pdf/10.1145/3543507.3583512"""
    # Generate uniform unit vectors
    noise = np.random.uniform(size=train_embeddings.shape)
    noise = noise / np.linalg.norm(noise, axis=1)[:, None]

    # Generate gamma distributed scale factors
    scale = np.random.gamma(
        shape=train_embeddings.shape[1],
        scale=1 / epsilon,
        size=train_embeddings.shape[0],
    )

    # Scale the noise by the scale factors
    noise = noise * scale[:, None]

    # Add the noise to the train embeddings
    train_embeddings = train_embeddings + noise
    return train_embeddings


def build_index(train_embeddings):
    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(train_embeddings)
    return index


def test(index, train_labels):
    # Load test embeddings
    test_embeddings, test_labels = load_embeddings(split="test")

    # Define a query example
    k = 11  # Number of nearest neighbors to retrieve
    n_corrects = 0  # List to store the results

    for query_embedding, query_label in zip(test_embeddings, test_labels):
        # Perform a k-nearest neighbors search
        distances, indices = index.search(query_embedding.reshape(1, -1), k)

        # Get the labels of the nearest neighbors
        neighbor_labels = [train_labels[idx] for idx in indices[0]]

        # Find the proposed label based on majority vote
        proposed_label = max(set(neighbor_labels), key=neighbor_labels.count)

        # Check if the proposed label is correct
        n_corrects += int(query_label == proposed_label)

    accuracy = n_corrects / len(test_embeddings)
    return accuracy


def run(epsilon):
    # Load train embeddings
    train_embeddings, train_labels = load_embeddings(split="train")
    # Add noise to train embeddings
    train_embeddings = add_noise_to_train_embeddings(train_embeddings, epsilon)
    # Build index
    index = build_index(train_embeddings)
    # Test
    accuracy = test(index, train_labels)
    return accuracy


def experiment():
    epsilons = range(1, 16)
    for epsilon in epsilons:
        accuracy = run(epsilon=epsilon)
        print(f"Epsilon: {epsilon}, Accuracy: {accuracy}")


if __name__ == "__main__":
    # run(epsilon=3.0)
    experiment()
