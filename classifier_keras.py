from keras.models import Sequential
# from keras.layers import Dense, Input, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU

INPUT_LENGTH = 30  # length of feature vector
OUTPUT_LENGTH = 150  # number of different users to classify

NUM_HIDDEN = 128  # number of hidden units in LSTM


def build_model(dropout_rate):
    """
    Builds classifier model (CNN + RNN)
    """

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(INPUT_LENGTH, 1)))
    model.add(MaxPooling1D())
    model.add(GRU(units=OUTPUT_LENGTH, activation="softmax"))

    return model


def load_data():
    """
    Loads training data.
    """
    X = None
    y = None

    return X, y


def next_batch(X, y):
    """
    Returns the next batch of training data.
    """
    X_batch = X
    y_batch = y

    return X_batch, y_batch


def main():

    # Hyperparameters
    training_epochs = 10
    dropout_rate = 0.1
    batch_size = 32

    # Load data
    X_train, y_train = load_data()

    # Build model
    model = build_model(dropout_rate)

    # Train model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=training_epochs)


if __name__ == "__main__":
    main()
