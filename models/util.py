import os
import random

import numpy as np
from tqdm import tqdm


def shuffle_data(X, y):
    """
    Shuffles the data in X, y with the same permutation.
    """

    n_examples = X.shape[0]

    perm = np.random.permutation(n_examples)
    X = X[perm, :, :]
    y = y[perm, :]

    return X, y


def split_data(X, y, train_frac, valid_frac, test_frac):
    """
    Splits data into train/valid/test-sets according to the specified fractions.
    """

    np.random.seed(1)

    assert train_frac + valid_frac + test_frac == 1, "Train/valid/test data fractions do not sum to one"

    n_examples = X.shape[0]

    # Shuffle
    X, y = shuffle_data(X, y)

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


def split_on_users(X, y, n_valid_users, add_other=False, n_invalid_users=None):
    """
    If add_other is False (default):

    Splits the given dataset into two sets:
    X_valid, y_valid - Data from the set of n_valid_users that are authorized.
    X_unknown, y_unknown - Data from the remaining set of users

    Data is relabeled as one-hot for n_valid_users

    -----------------------------------------------------------------------------------

    If add_other is True (used to split data for "valid-plus-others" approach):

    n_invalid_users must be specified.

    Splits the given dataset into three sets:
    X_valid, y_valid - Data from the set of n_valid_users that are authorized.
    X_invalid, y_invalid - Data from a set of n_invalid_users that are known to be unauthorized.
    X_unknown, y_unknown - Data from users that are unauthorized, but never gets seen during training.

    Data is relabeled as one-hot for n_valid_users + 1
    """

    n_examples, n_users = y.shape

    if add_other:
        assert n_invalid_users is not None, "Argument n_invalid_users must be specified when add_other is True."
        assert n_valid_users + n_invalid_users <= n_users, "Number of valid/invalid users specified exceeds the total number of users."
    else:
        assert n_valid_users <= n_users, "Number of valid users specified exceeds the total number of users."

    valid_users = random.sample(range(n_users), k=n_valid_users)

    if add_other:
        remaining_users = [user for user in range(n_users) if user not in valid_users]
        invalid_users = random.sample(remaining_users, k=n_invalid_users)

    X_valid, y_valid = [], []
    X_unknown, y_unknown = [], []
    if add_other:
        X_invalid, y_invalid = [], []

    for i in range(n_examples):
        user = np.asscalar(np.where(y[i, :] == 1)[0])
        if user in valid_users:
            X_valid.append(X[i, :])
            y_valid.append(valid_users.index(user))
        elif add_other:
            if user in invalid_users:
                X_invalid.append(X[i, :])
                y_invalid.append(n_valid_users)
            else:
                X_unknown.append(X[i, :])
                y_unknown.append(n_valid_users)
        else:
            X_unknown.append(X[i, :])
            y_unknown.append(-1)

    X_valid, y_valid = np.asarray(X_valid), np.asarray(y_valid)
    X_unknown, y_unknown = np.asarray(X_unknown), np.asarray(y_unknown)
    if add_other:
        X_invalid, y_invalid = np.asarray(X_invalid), np.asarray(y_invalid)

    if add_other:
        y_valid = index_to_one_hot(y_valid, n_valid_users + 1)
        y_unknown = index_to_one_hot(y_unknown, n_valid_users + 1)
        y_invalid = index_to_one_hot(y_invalid, n_valid_users + 1)
    else:
        y_valid = index_to_one_hot(y_valid, n_valid_users)
        y_unknown = index_to_one_hot(y_unknown, n_valid_users)

    if add_other:
        return X_valid, y_valid, X_invalid, y_invalid, X_unknown, y_unknown

    return X_valid, y_valid, X_unknown, y_unknown


def index_to_one_hot(y, n_classes):
    """
    Converts a list of indices to one-hot encoding.
    Example: y = [1, 0, 3] => np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    If a label is -1 (unknown), its one-hot enoding becomes [-1, ..., -1]
    """

    minus_one = np.where(y == -1)
    y = y.reshape(-1)
    one_hot = np.eye(n_classes)[y]
    one_hot[minus_one, :] = -np.ones((n_classes,))

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
