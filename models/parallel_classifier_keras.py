import argparse
import numpy as np
from tqdm import tqdm

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

    model = Model(inputs=[inp1, inp2], outputs=op)

    print(model.summary())

    return model


def compute_FAR_FRR(trained_model, X_test_1, X_test_2, y_test):
    """
    Computes the FAR and FRR of trained_model on the given test set.
    """

    n_examples = X_test_1.shape[0]
    n_valid_users = y_test.shape[1]

    n_imposter_tries = 0
    n_valid_tries = 0
    FA_errors = 0
    FR_errors = 0

    # Let every person claim to be every user
    for i in tqdm(range(n_examples)):
        input_1 = X_test_1[i, :, :]
        input_1 = input_1.reshape((1,) + input_1.shape)
        input_2 = X_test_2[i, :, :]
        input_2 = input_2.reshape((1,) + input_2.shape)

        y_pred = trained_model.predict([input_1, input_2])
        y_true = y_test[i, :]

        for id in range(n_valid_users):

            # If valid user
            if y_true[0] != -1 and np.argmax(y_true) == id:
                n_valid_tries += 1
                if np.argmax(y_pred) != id:
                    FR_errors += 1

            # If imposter
            else:
                n_imposter_tries += 1
                if np.argmax(y_pred) != n_valid_users:
                    FA_errors += 1

    FAR = float(FA_errors)/n_imposter_tries
    FRR = float(FR_errors)/n_valid_tries

    return FAR, FRR


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", metavar="PATH", default=None, help="Path to read processed digraph data from.")
    parser.add_argument("-n", "--n_users", metavar="N_USERS", default=20, help="Number of users to enroll.", type=int)
    args = parser.parse_args()

    # Load all data
    X, y = util.load_data(args.data_path, EXAMPLE_LENGTH)

    # Split into random set of valid (v) and unknown (u) users (and relabel accordingly)
    X_v, y_v, X_u, y_u = util.split_on_users(X, y, n_valid_users=args.n_users)

    # Split into train/valid/test
    X_train, y_train, X_valid, y_valid, X_test_v, y_test_v = util.split_data(X_v, y_v, train_frac=0.8, valid_frac=0.1, test_frac=0.1)
    X_test_u, y_test_u = X_u, y_u

    # Concatenate test data
    X_test = np.vstack((X_test_v, X_test_u))
    y_test = np.vstack((y_test_v, y_test_u))

    # Save non-shuffled data
    X_train_non_shuffled, y_train_non_shuffled = X_train, y_train
    X_valid_non_shuffled, y_valid_non_shuffled = X_valid, y_valid
    X_test_non_shuffled, y_test_non_shuffled = X_test, y_test

    # Shuffle the data
    X_train, y_train = util.shuffle_data(X_train, y_train)
    X_valid, y_valid = util.shuffle_data(X_valid, y_valid)
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

    # Outer model

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

    # Split into sub-inputs
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
    print("Evaluating model...")
    loss, accuracy = model.evaluate([X_test1, X_test2], y_test, verbose=1)
    FAR, FRR = compute_FAR_FRR(model, X_test1, X_test2, y_test)

    print("\n---- Test Results ----")
    print("Test loss = {}, Test accuracy = {}".format(loss, accuracy))
    print("FAR = {}, FRR = {}".format(FAR, FRR))


if __name__ == "__main__":
    main()
