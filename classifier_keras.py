import signal
import os

import argparse
import numpy as np
from tqdm import tqdm

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import Callback, ModelCheckpoint

import util

# Hyperparameters
EPOCHS = 100
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

# Global variables
stop_flag = False  # Flag to indicate that training was terminated early
training_complete = False  # Flag to indicate that training is complete


class TerminateOnFlag(Callback):
    """
    Callback that terminates training at the end of an epoch if stop_flag is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        if stop_flag:
            self.model.stop_training = True


def handler(signum, frame):
    """
    Flags stop_flag if CTRL-C is received.
    """
    global training_complete

    if not training_complete:
        print('\nCTRL+C signal received. Training will finish after current batch.')
        global stop_flag
        stop_flag = True
    else:
        exit()


def build_model(input_shape, n_classes):
    """
    Builds classifier model (CNN + Dense)
    """

    model = Sequential()

    # Convolutional layers
    model.add(Conv1D(128, 2, activation="relu", input_shape=input_shape))
    model.add(Dropout(rate=DROPOUT_RATE))
    model.add(Conv1D(128, 2, activation="relu"))
    model.add(Dropout(rate=DROPOUT_RATE))
    model.add(Conv1D(128, 2, activation="relu"))
    model.add(Dropout(rate=DROPOUT_RATE))
    model.add(Conv1D(64, 2, activation="relu"))
    model.add(Dropout(rate=DROPOUT_RATE))

    model.add(Flatten())

    # Dense layers
    model.add(Dense(n_classes*8, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(n_classes*4, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    return model


def compute_FAR_FRR(trained_model, X_test, y_test):
    """
    Computes the FAR and FRR of trained_model on the given test set.

    Assumes label is [-1, ..., -1] is user is unknown (i.e. unauthorized)
    """

    n_examples = X_test.shape[0]
    n_valid_users = y_test.shape[1]

    n_imposter_tries = 0
    n_valid_tries = 0
    FA_errors = 0
    FR_errors = 0

    # Let every person claim to be every user
    for i in tqdm(range(n_examples)):

        y_pred = trained_model.predict(X_test[i, :, :].reshape((1,) + X_test[i, :, :].shape))
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
                if np.argmax(y_pred) == id:
                    FA_errors += 1

    FAR = float(FA_errors)/n_imposter_tries
    FRR = float(FR_errors)/n_valid_tries

    return FAR, FRR


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="data_path", metavar="DATA_PATH", help="Path to read examples from.")
    parser.add_argument(dest="model_path", metavar="MODEL_PATH", help="Path to save trained model to.")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        response = input("Model path does not exist. Create it? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()
        else:
            os.makedirs(args.model_path)

    # Load all data
    X_train, y_train, X_valid, y_valid, X_test, y_test = util.load_examples(args.data_path)

    # Shuffle the data
    X_train, y_train = util.shuffle_data(X_train, y_train)
    X_valid, y_valid = util.shuffle_data(X_valid, y_valid)
    X_test, y_test = util.shuffle_data(X_test, y_test)

    # Build model
    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1]
    model = build_model(input_shape, n_classes)

    # Compile model
    adam_optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])

    # Train model
    signal.signal(signal.SIGINT, handler)
    terminate_on_flag = TerminateOnFlag()  # Terminate training if CTRL+C
    model_checkpoint = ModelCheckpoint(args.model_path + str(n_classes) + "_class_model_{epoch:02d}_{val_loss:.2f}.hdf5",
                                       monitor="val_loss", save_best_only=True, verbose=1, period=5)  # Save model every 5 epochs

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=BATCH_SIZE,
              epochs=EPOCHS, callbacks=[terminate_on_flag, model_checkpoint])

    global training_complete
    training_complete = True

    # Test model
    print("Evaluating model...")
    FAR, FRR = compute_FAR_FRR(model, X_test, y_test)
    print("\n---- Test Results ----")
    print("FAR = {}, FRR = {}".format(FAR, FRR))


if __name__ == "__main__":
    main()
