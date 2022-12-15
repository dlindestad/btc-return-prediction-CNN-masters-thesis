import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_model_loss_training(model):
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch

    fig, ax = plt.subplots(1, figsize=(8, 6))
    num_epochs = model_history.shape[0]

    ax.plot(np.arange(0, num_epochs), model_history["loss"], label="Training loss")
    ax.plot(
        np.arange(0, num_epochs), model_history["val_loss"], label="Validation loss"
    )
    ax.legend()
    plt.yscale(value="log")
    plt.tight_layout()
    plt.show()


def plot_prices_entire(training, validation, predictions):
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close price USD", fontsize=18)
    plt.plot(training["price"])
    plt.plot(validation["price"])
    plt.plot(predictions["price"])
    plt.legend(["train", "val", "predictions"], loc="lower right")
    plt.show()


def plot_prices_per_return(training, validation):
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close price USD", fontsize=18)
    plt.plot(training["price"])
    plt.plot(validation[["price", "predictions"]])
    plt.legend(["train", "val", "predictions"], loc="lower right")
    plt.show()


def plot_returns(training, validation, predictions):
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Return, USD", fontsize=18)
    plt.plot(training["return"])
    plt.plot(validation["return"])
    plt.plot(predictions["predictions"])
    plt.legend(["train", "val", "predictions"], loc="lower right")
    plt.show()
