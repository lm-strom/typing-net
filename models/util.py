import os
import numpy as np
from tqdm import tqdm


def split_data(X, y, train_frac, valid_frac, test_frac):
    """
    Splits data into train/valid/test-sets according to the specified fractions.
    """

    np.random.seed(1)

    assert train_frac + valid_frac + test_frac == 1, "Train/valid/test data fractions do not sum to one"

    n_examples = X.shape[0]

    # Shuffle
    perm = np.random.permutation(n_examples)
    X = X[perm, :, :]
    y = y[perm, :]

    # Split
    ind_1 = int(np.round(train_frac*n_examples))
    ind_2 = int(np.round(ind_1 + valid_frac*n_examples))

    X_train = X[0:ind_1, :, :]
    y_train = y[0:ind_1, :]
    X_valid = X[ind_1:ind_2, :, :]
    y_valid = y[ind_1:ind_2, :]
    X_test = X[ind_2:, :, :]
    y_test = y[ind_2:, :]

    assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == n_examples, "Data split failed"

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def load_data(data_path, example_length):
    """
    Loads all data in data_path.
    Creates examples of length example_length.

    Returns:
    Matrix X of shape (#examples, example_length, feature_length)
    Matrix y of shape (#examples, #users)
    """

    X = []
    y = []

    n_users = len(os.listdir(data_path))

    print("Loading data...")
    for i, user_file_name in tqdm(enumerate(os.listdir(data_path))):
        if user_file_name[0] == ".":
            continue
        with open(data_path + user_file_name, "r") as user_file:
            example = []
            for line in user_file:
                feature = tuple(map(int, line.split()))
                example.append(feature)

                if len(example) == example_length:
                    X.append(example)
                    y.append(i)
                    example = []

    X = np.asarray(X)
    y = np.asarray(y)

    y = index_to_one_hot(y, n_users)

    return X, y


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
