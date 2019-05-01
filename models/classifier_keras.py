import os
import numpy as np
from tqdm import tqdm

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU

DATA_PATH = "/Users/Hannes/Documents/typing-net/data/processed_data/"

# Constants
FEATURE_LENGTH = 6

# Hyperparameters
EXAMPLE_LENGTH = 20

EPOCHS = 100
DROPOUT_RATE = 0.1  # currently not used
BATCH_SIZE = 32
LEARNING_RATE = 2e-4


def build_model(input_shape, n_classes):
	"""
	Builds classifier model (CNN + RNN)
	"""

	model = Sequential()
	model.add(Conv1D(32, 2, activation="sigmoid", input_shape=input_shape))
	model.add(Conv1D(32, 2, activation="relu"))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(n_classes*8, activation="sigmoid"))
	model.add(Dense(n_classes*4, activation="relu"))
	model.add(Dense(n_classes, activation="softmax"))
	# model.add(GRU(units=n_classes, activation="softmax"))

	print(model.summary())

	return model


def index_to_one_hot(y, n_classes):
	"""
	Converts a list of indices to one-hot encoding.
	Example: y = [1, 0, 3] => np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
	"""
	y = y.reshape(-1)
	one_hot = np.eye(n_classes)[y]

	return one_hot


def one_hot_to_index(y):
	"""
	Converts numpy array of one-hot encodings to list of indices.
	Example: y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) => [1, 0, 3]
	"""
	indices = []
	for num in y:
		if np.nonzero(num)[0].size == 0:
			indices.append(-1)
		else:
			indices.append(np.nonzero(num)[0][0])

	return indices


def load_data():
	"""
	Loads all data. Creates examples of length EXAMPLE_LENGTH.
	Returns:
	Matrix X of shape (#examples, example_len, feature_length)
	Matrix y of shape (#examples, #users)
	"""

	X = []
	y = []

	n_users = len(os.listdir(DATA_PATH))

	print("Loading data...")
	for i, user_file_name in tqdm(enumerate(os.listdir(DATA_PATH))):
		if user_file_name[0] == ".":
			continue
		with open(DATA_PATH + user_file_name, "r") as user_file:
			example = []
			for line in user_file:
				feature = tuple(map(int, line.split()))
				example.append(feature)

				if len(example) == EXAMPLE_LENGTH:
					X.append(example)
					y.append(i)
					example = []

	X = np.asarray(X)
	y = np.asarray(y)

	y = index_to_one_hot(y, n_users)

	return X, y


def split_data(X, y, train_frac, valid_frac, test_frac):

	assert train_frac + valid_frac + test_frac == 1, "Train/valid/test data fractions do not sum to one"

	n_examples = X.shape[0]

	# Shuffle
	perm = np.random.permutation(n_examples)
	X = X[perm, :, :]
	y = y[perm, :]

	# Split
	ind_1 = int(np.round(train_frac*n_examples))
	ind_2 = int(np.round(ind_1 + valid_frac*n_examples))

	X_train = X[0:ind_1, :,:]
	y_train = y[0:ind_1, :]
	X_valid = X[ind_1:ind_2, :,:]
	y_valid = y[ind_1:ind_2, :]
	X_test = X[ind_2:, :,:]
	y_test = y[ind_2:, :]


	assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == n_examples, "Data split failed"

	return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def next_batch(X, y):
	"""
	Returns the next batch of training data.
	"""
	X_batch = X
	y_batch = y

	return X_batch, y_batch


def main():

	# Load all data
	X, y = load_data()

	# Split that shit
	X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y, train_frac=0.8, valid_frac=0.1, test_frac=0.1)

	# Build model
	input_shape = X_train.shape[1:]
	n_classes = y_train.shape[1]
	model = build_model(input_shape, n_classes)

	# Train model
	adam_optimizer = optimizers.Adam(lr=LEARNING_RATE)
	model.compile(loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])
	model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS)

	# Test model
	loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

	print("\n---- Test Results ----")
	print("Test loss = {}, Test accuracy = {}".format(loss, accuracy))


if __name__ == "__main__":
	main()
