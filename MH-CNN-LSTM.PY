import numpy as np
import pandas as pd
import tensorflow as tf
from functions import *
from import_data import data_scaled, data_scalers, data_raw


WINDOW = 150
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

    # Define the input layer
    inputs = tf.keras.Input(shape=[WINDOW, n_features])

    head_list = []
    for i in range(0, n_features):
        conv_layer_head = tf.keras.layers.Conv1D(
            filters=4, kernel_size=7, activation="relu"
        )(inputs)
        conv_layer_head_2 = tf.keras.layers.Conv1D(
            filters=6, kernel_size=11, activation="relu"
        )(conv_layer_head)
        conv_layer_flatten = tf.keras.layers.Flatten()(conv_layer_head_2)
        head_list.append(conv_layer_flatten)

    concat_cnn = tf.keras.layers.Concatenate(axis=1)(head_list)
    reshape = tf.keras.layers.Reshape((head_list[0].shape[1], n_features))(concat_cnn)
    lstm = tf.keras.layers.LSTM(100, activation="relu", return_sequences=True)(reshape)
    dropout = tf.keras.layers.Dropout(0.2)(lstm)
    lstm_2 = tf.keras.layers.LSTM(100, activation="relu", return_sequences=False)(
        dropout
    )
    dropout2 = tf.keras.layers.Dropout(0.2)(lstm_2)
    outputs = tf.keras.layers.Dense(1, activation="linear")(dropout2)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file="Plots/model_shape_MH-CNN-LSTM.png", show_shapes=True
    )
    #
    # print(model.summary())
    # stop early if model is not improving for patience=n epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=100, restore_best_weights=True
    )
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["mse", "mae"],
    )
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
        "MH_CNN_LSTM",
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
