import tensorflow as tf
from utils import get_data, standardize
import argparse
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Arguments for multi-label prediction on surveillance dataset.')
    # Dataset arguments
    parser.add_argument("-s", "--size", default=992, type=int,
                        help="Training data size")
    parser.add_argument("-c", "--cardinality", default=365, type=int,
                        help="Cardinality of possible labels.")
    # Model arguments
    parser.add_argument("--hidden_units", default="100", type=str,
                        help="Hidden units separated by ','")
    # Training arguments
    parser.add_argument("--batch_size", default=100, type=int,
                        help="Batch size for training")
    parser.add_argument("--lr", default=1e-2, type=float,
                        help="Learning rate.")
    parser.add_argument("-e", "--epochs", default=1000, type=int,
                        help="Number of epochs")
    parser.add_argument("--is_privacy_preserve", action="store_true",
                        help="Whether train the model by DPSGD")
    # Differential privacy arguments
    parser.add_argument("--l2_norm_clip", default=1.0, type=float,
                        help="L2 norm clip value")
    parser.add_argument("--noise_multiplier", default=0.5, type=float,
                        help="Noise mulitplier (gaussian delta)")
    parser.add_argument("--num_microbatches", default=1, type=int,
                        help="Number of microbatches")
    args = parser.parse_args()
    return args


def get_model(out_units, hidden_units: str):
    hidden_units = [int(s) for s in hidden_units.split(',')]
    model = tf.keras.models.Sequential()
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units=units, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_units))
    return model


def get_optimizer(args):
    if args.is_privacy_preserve:
        return tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            num_microbatches=args.num_microbatches,
            learning_rate=args.lr)
    else:
        return tf.keras.optimizers.Adam(learning_rate=args.lr)


def run(args):
    embeddings, labels = get_data(args.size, args.cardinality)
    embeddings = standardize(embeddings)
    model = get_model(args.cardinality, args.hidden_units)
    opt = get_optimizer(args)
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
    model.compile(optimizer=opt, loss=loss_func, metrics=metrics)

    model.fit(embeddings, labels, batch_size=args.batch_size, epochs=args.epochs)

    # Print model and privacy budget
    print(model.summary())
    print(args)
    if args.is_privacy_preserve:
        eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=embeddings.shape[0],
            batch_size=args.batch_size,
            noise_multiplier=args.noise_multiplier,
            epochs=args.epochs,
            delta=1e-5
        )
        print(f'Privacy budget is : {eps:.3f}')


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
