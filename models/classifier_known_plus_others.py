import argparse
import numpy as np

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D
from keras.layers import Input, Concatenate

import util

# Constants
FEATURE_LENGTH = 6
N_VALID_USERS = 20  # number of authorized users
N_INVALID_USERS = 20  # number of unauthorized users considered in training

# Hyperparameters
EXAMPLE_LENGTH = 18

EPOCHS_SUBMODEL = 40
EPOCHS = 100
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 3e-4


def build_submodel(input_shape, n_classes):
    """
    Builds classifier model (CNN + Dense)
    """

    model = Sequential()

    # Convolutional layers
    model.add(Conv1D(128, 2, activation="sigmoid", input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv1D(128, 2, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv1D(128, 2, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv1D(64, 2, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Flatten())

    # Dense layers
    model.add(Dense(n_classes*8, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(n_classes*4, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    return model


def build_model(submodel, weights, input_shape, n_classes):
    """
    Builds classifier model (CNN + RNN)
    """

    print(input_shape)

    inp1 = Input(shape=input_shape)
    inp2 = Input(shape=input_shape)

    subnet1 = submodel(inputs=inp1)
    subnet2 = submodel(inputs=inp2)

    mrg = Concatenate()([subnet1, subnet2])
    dense = Dense(n_classes*4, activation="relu")(mrg)
    dense = Dense(n_classes*4, activation="relu")(dense)
    op = Dense(n_classes, activation="softmax")(dense)

    model = Model(input=[inp1, inp2], output=op)

    print(model.summary())

    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", metavar="PATH", default=None, help="Path to read processed digraph data from.")
    args = parser.parse_args()

    # Load all data
    X, y = util.load_data(args.data_path, EXAMPLE_LENGTH)

    # Split into random set of valid (v), invalid (i) and unknown (u) users (and relabel accordingly)
    X_v, y_v, X_i, y_i, X_u, y_u = util.split_on_users(X, y, n_valid_users=N_VALID_USERS, n_invalid_users=N_INVALID_USERS)

    # Split into train/valid/test
    X_train_v, y_train_v, X_valid_v, y_valid_v, X_test_v, y_test_v = util.split_data(X_v, y_v, train_frac=0.6, valid_frac=0.2, test_frac=0.2)
    X_train_i, y_train_i, X_valid_i, y_valid_i, X_test_i, y_test_i = util.split_data(X_i, y_i, train_frac=0.8, valid_frac=0.1, test_frac=0.1)
    _, _, _, _, X_test_u, y_test_u = util.split_data(X_u, y_u, train_frac=0, valid_frac=0, test_frac=1)

    # Concatenate and shuffle the two data sets
    X_train = np.vstack((X_train_v, X_train_i))
    y_train = np.vstack((y_train_v, y_train_i))
    X_train_non_shuffled, y_train_non_shuffled = X_train, y_train
    X_train, y_train = util.shuffle_data(X_train, y_train)

    X_valid = np.vstack((X_valid_v, X_valid_i))
    y_valid = np.vstack((y_valid_v, y_valid_i))
    X_valid_non_shuffled, y_valid_non_shuffled = X_valid, y_valid
    X_valid, y_valid = util.shuffle_data(X_valid, y_valid)

    X_test = np.vstack((X_test_v, X_test_i, X_test_u))
    y_test = np.vstack((y_test_v, y_test_i, y_test_u))
    X_test_non_shuffled, y_test_non_shuffled = X_test, y_test
    X_test, y_test = util.shuffle_data(X_test, y_test)

    # Build model
    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1]
    submodel = build_submodel(input_shape, n_classes)

    # Train model
    adam_optimizer = optimizers.Adam(lr=LEARNING_RATE)
    submodel.compile(loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])
    submodel.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS_SUBMODEL)

    weights = submodel.get_weights()


    # ------------------------

    # Reshape to double example_length
    X_shape = X_train_non_shuffled.shape
    if X_shape[0] % 2 != 0:
        X_train_non_shuffled = np.delete(X_train_non_shuffled, -1, axis=0)
        y_train_non_shuffled = np.delete(y_train_non_shuffled, -1, axis=0)
    X_train = X_train_non_shuffled.reshape(X_shape[0]//2, 2*X_shape[1], X_shape[2])
    y_train = y_train_non_shuffled[::2]

    X_shape = X_valid_non_shuffled.shape
    if X_shape[0] % 2 != 0:
        X_valid_non_shuffled = np.delete(X_valid_non_shuffled, -1, axis=0)
        y_valid_non_shuffled = np.delete(y_valid_non_shuffled, -1, axis=0)
    X_valid = X_valid_non_shuffled.reshape(X_shape[0]//2, 2*X_shape[1], X_shape[2])
    y_valid = y_valid_non_shuffled[::2]

    X_shape = X_test_non_shuffled.shape
    if X_shape[0] % 2 != 0:
        X_test_non_shuffled = np.delete(X_test_non_shuffled, -1, axis=0)
        y_test_non_shuffled = np.delete(y_test_non_shuffled, -1, axis=0)
    X_test = X_test_non_shuffled.reshape(X_shape[0]//2, 2*X_shape[1], X_shape[2])
    y_test = y_test_non_shuffled[::2]

    X_train1 = X_train[:, :EXAMPLE_LENGTH, :]
    X_train2 = X_train[:, EXAMPLE_LENGTH:, :]

    X_valid1 = X_valid[:, :EXAMPLE_LENGTH, :]
    X_valid2 = X_valid[:, EXAMPLE_LENGTH:, :]

    X_test1 = X_test[:, :EXAMPLE_LENGTH, :]
    X_test2 = X_test[:, EXAMPLE_LENGTH:, :]

    # Build model
    model = build_model(submodel, weights, input_shape, n_classes)

    model.layers[2].trainable = False

    # Train model
    adam_optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])
    model.fit([X_train1, X_train2], y_train, validation_data=([X_valid1, X_valid2], y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Test model
    loss, accuracy = model.evaluate([X_test1, X_test2], y_test, verbose=1)

    print("\n---- Test Results ----")
    print("Test loss = {}, Test accuracy = {}".format(loss, accuracy))


if __name__ == "__main__":
    main()
