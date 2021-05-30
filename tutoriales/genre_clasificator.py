import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout

DATASET_PATH = "data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def plot_history(history):
    fig, ax = plt.subplots(2)

    # accuracy subplot
    ax[0].plot(history.history["accuracy"], label="train accuracy")
    ax[0].plot(history.history["val_accuracy"], label="test accuracy")
    ax[0].set_ylabel("Accurracy")
    ax[0].legend(loc="lower right")
    ax[0].set_title("Accuracy eval")

    # error subplot
    ax[1].plot(history.history["loss"], label="train error")
    ax[1].plot(history.history["val_loss"], label="test error")
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3
    )

    # build network architectur
    model = keras.Sequential(
        [
            # input layer
            keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
            # 1st hidden layer
            keras.layers.Dense(
                512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # 2nd hidden layer
            keras.layers.Dense(
                256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # 3d hidden layer
            keras.layers.Dense(
                64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # output layers
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    # train network
    history = model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=50,
        batch_size=32,
    )

    # plot accuracy and error over the epochs
    plot_history(history)
