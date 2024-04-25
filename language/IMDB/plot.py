import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_results():
    # Load the data
    data = pd.read_csv("BERTvsBiLSTM.csv")

    # Plot settings
    plt.figure(figsize=(8, 5))
    plt.rc("font", size=15)  # controls default text sizes
    plt.tight_layout()

    # Create a plot

    sns.lineplot(
        x="privacy budget",
        y="accuracy",
        hue="method",
        data=data,
    )

    # Save the plot
    plt.savefig("BERTvsBiLSTM.png")


plot_results()
