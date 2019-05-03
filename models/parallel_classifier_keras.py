import argparse

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D
from keras.layers import Input, Concatenate

import util

# Constants
FEATURE_LENGTH = 6

# Hyperparameters
EXAMPLE_LENGTH = 18

EPOCHS_SUBMODEL = 80
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

    # Split into train/valid/test
    X_train, y_train, X_valid, y_valid, X_test, y_test = util.split_data(X, y, train_frac=0.8, valid_frac=0.1, test_frac=0.1)

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

    # Load all data
    X, y = util.load_data(args.data_path, EXAMPLE_LENGTH*2)

    # Split into train/valid/dev
    X_train, y_train, X_valid, y_valid, X_test, y_test = util.split_data(X, y, train_frac=0.8, valid_frac=0.1, test_frac=0.1)

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
