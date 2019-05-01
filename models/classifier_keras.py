import os
import numpy as np
from tqdm import tqdm

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Input, Concatenate
from keras.layers import GRU

DATA_PATH = "/Users/Hannes/Downloads/typing-net/data/processed_data/"

# Constants
FEATURE_LENGTH = 6

# Hyperparameters
EXAMPLE_LENGTH = 18

EPOCHS = 100
DROPOUT_RATE = 0.1  # currently not used
BATCH_SIZE = 32
LEARNING_RATE = 3e-4


def build_model(input_shape, n_classes):
	"""
	Builds classifier model (CNN + RNN)
	"""

	new_input_shape = (int(input_shape[0]/2), input_shape[1])

	inp1 = Input(shape=new_input_shape)
	inp2 = Input(shape=new_input_shape)

	conv1_1 = Conv1D(32, 2, activation="sigmoid", input_shape=input_shape)(inp1)
	conv2_1 = Conv1D(32, 2, activation="sigmoid", input_shape=input_shape)(inp2)
	conv1_2 = Conv1D(32, 2, activation="relu", input_shape=input_shape)(conv1_1)
	conv2_2 = Conv1D(32, 2, activation="relu", input_shape=input_shape)(conv2_1)
	maxp1 = MaxPooling1D()(conv1_2)
	maxp2 = MaxPooling1D()(conv2_2)
	flatten1 = Flatten()(maxp1)
	flatten2 = Flatten()(maxp2)
	dense1_1 = Dense(n_classes*8, activation="sigmoid")(flatten1)
	dense2_1 = Dense(n_classes*8, activation="sigmoid")(flatten2)
	dense1_2 = Dense(n_classes*4, activation="relu")(dense1_1)
	dense2_2 = Dense(n_classes*4, activation="relu")(dense2_1)
	dense1_3 = Dense(n_classes, activation="softmax")(dense1_2)
	dense2_3 = Dense(n_classes, activation="softmax")(dense2_2)

	mrg = Concatenate()([dense1_3, dense2_3])
	dense = Dense(n_classes, activation="relu")(mrg)
	op = Dense(n_classes, activation="softmax")(dense)

	model = Model(input=[inp1, inp2], output=op)

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

	X_train1 = X_train[:,:int(int(EXAMPLE_LENGTH/2)),:]
	X_train2 = X_train[:,int(int(EXAMPLE_LENGTH/2)):,:]

	X_valid1 = X_valid[:,:int(int(EXAMPLE_LENGTH/2)),:]
	X_valid2 = X_valid[:,int(EXAMPLE_LENGTH/2):,:]

	X_test1 = X_test[:,:int(EXAMPLE_LENGTH/2),:]
	X_test2 = X_test[:,int(EXAMPLE_LENGTH/2):,:]


	# Build model
	input_shape = X_train.shape[1:]
	n_classes = y_train.shape[1]
	model = build_model(input_shape, n_classes)

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
