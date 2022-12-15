import numpy as np
import pandas as pd
import tensorflow as tf
from functions import *
from import_data import data_scaled, data_scalers, data_raw


WINDOW = 50
HORIZON = 1
SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 32


def main():
    ################### Data pre processing ######################
    data = data_scaled
    x_train, y_train, x_test, y_test = create_train_test(
        data,
        {"btc_logreturn": data["btc_logreturn"]},
        window=WINDOW,
        horizon=HORIZON,
        split=SPLIT,
    )

    n_features = len(data.keys())
    print(x_test.shape)

    print(y_test[:, 0, 0].shape)
    print(y_test[1:4, 0, 0])

    # define the CNN model
    # define the input tensor
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[WINDOW, n_features]),
            # define the convolutional layers
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=3, padding="same", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, padding="same", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(
                filters=128, kernel_size=3, padding="same", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # flatten the output of the convolutional layers
            tf.keras.layers.Flatten(),
            # define the fully-connected layers
            tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
            # define the output layer
            tf.keras.layers.Dense(units=1),
        ],
    )
    # model.build(input_shape=x_train.shape)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file="Plots/model_shape_CNN.png", show_shapes=True
    )
    #
    # print(model.summary())
    # stop early if model is not improving for patience=n epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=15, restore_best_weights=True
    )
    model.compile(loss="mae", optimizer="adam", metrics=["mse", "mae"])
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[es_callback],
        shuffle=False,
    )

    epoch_count = len(model.history.epoch)
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch
    num_epochs = model_history.shape[0]
    # plot_model_loss_training(model)

    # get the models predicted values
    # pred_train = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_train))
    pred_test = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_test))
    y_test_unscaled = data_scalers["btc_logreturn"].inverse_transform(y_test[:, :, 0])
    model_stats(model.name, pred_test, x_test, y_test_unscaled, model, epoch_count)
    print_model_statistics(
        model_statistics(pred_test, x_test, y_test, model), x_test, y_test, model
    )

    save_loss_history(
        num_epochs, model_history["loss"], model_history["val_loss"], "last_model"
    )
    main_plot(
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
    )
    pred_plot(
        "CNN",
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
    )
    return


if __name__ == "__main__":
    main()
