import sys
import os
import argparse
import random

import numpy as np
from tqdm import tqdm
import h5py

assert os.getcwd().split("/")[-1] == "typing-net", "Preprocessing scripts must run from typing-net/ directory."
sys.path.insert(0, 'models/')  # so that utils can be imported
import utils
import generate_triplets

# Constants
FEATURE_LENGTH = 5


def create_examples(input_path, data_file, example_length):
    """
    Loads all digraph data in input_path, creates examples of length example_length
    and corresponding labels.

    Returns:
    Matrix X of shape (#examples, example_length, feature_length)
    Matrix y of shape (#examples, #users)
    """

    n_users = len(os.listdir(input_path))

    X = []   # np.empty((0, example_length, FEATURE_LENGTH))
    y = []  # np.empty((0, 1))

    print("Generating examples...")
    for i, user_file_name in tqdm(enumerate(os.listdir(input_path))):
        if user_file_name[0] == ".":
            continue
        with open(input_path + user_file_name, "r") as user_file:
            example = []
            for line in user_file:
                feature = tuple(map(int, line.split()))
                example.append(feature)

                if len(example) == example_length:
                    X.append(np.asarray(example))
                    y.append(i)
                    example = []

    X = np.asarray(X)
    y = np.asarray(y)

    y = utils.index_to_one_hot(y, n_users)

    data_file.create_dataset("X_plain", data=X, maxshape=(None, example_length, FEATURE_LENGTH), dtype=float)
    data_file.create_dataset("y_plain", data=y, maxshape=(None, n_users), dtype=float)

    return X, y


def generate_examples_from_adjacents(X, y, dataset_name, data_file, step_size=1):
    """
    Parses through the data and generates (example_length - 1)//step_size
    additional examples from every pair of adjacent examples with the same label.
    The default step=1 generates *all* possible additional examples.

    The new examples are iteratively written to the dataset with dataset_name in data_file.
    """

    WRITE_CHUNK_SIZE = 1000

    n_examples = X.shape[0]
    example_length = X.shape[1]
    n_users = y.shape[1]

    assert step_size <= example_length - 1 and step_size > 0, "Invalid step size. Must have 0 < step <= example_length - 1"

    name_X = "X_" + dataset_name
    name_y = "y_" + dataset_name

    X_additional = np.empty((0, example_length, FEATURE_LENGTH))
    y_additional = np.empty((0, n_users))

    print("Augmenting to generate additional examples...")
    for i in tqdm(range(n_examples - 1)):

        # If not the same label: continue
        if np.any(y[i, :] != y[i + 1, :]):
            continue

        concat = np.vstack((X[i, :, :], X[i + 1, :, :]))

        for start in range(1, example_length, step_size):
            end = start + example_length

            new_example = np.expand_dims(concat[start:end, :], axis=0)
            new_label = np.expand_dims(y[i, :], axis=0)

            X_additional = np.append(X_additional, new_example, axis=0)
            y_additional = np.append(y_additional, new_label, axis=0)

        if X_additional.shape[0] >= WRITE_CHUNK_SIZE:
            data_file[name_X].resize(data_file[name_X].shape[0] + X_additional.shape[0], axis=0)
            data_file[name_X][-X_additional.shape[0]:] = X_additional

            data_file[name_y].resize(data_file[name_y].shape[0] + y_additional.shape[0], axis=0)
            data_file[name_y][-y_additional.shape[0]:] = y_additional

            X_additional = np.empty((0, example_length, FEATURE_LENGTH))
            y_additional = np.empty((0, n_users))


def split_all_users(X_data_name, y_data_name, output_name, data_file, append_randoms=False):
    """
    Splits the data into one dataset for each user. Writes each new dataset to the
    datafile with the names "X_user_{USER NUMBER}" and "y_user_{USER NUMBER}".

    Relabels every example with the integer 1, since the user identity can now be inferred
    from which dataset it is in.

    If append_randoms is True, an equal number of random examples from random other users
    are appended to each user's dataset, with the label 0.
    """

    n_examples = data_file[X_data_name].shape[0]
    n_users = data_file[y_data_name].shape[1]

    # Find which examples belong to which user
    print("Preparing split on users...")
    user_ind_dict = {j: [] for j in range(n_users)}
    for i in tqdm(range(n_examples)):
        j = utils.one_hot_to_index(data_file[y_data_name][i, :])
        user_ind_dict[j].append(i)

    # Extract the data for each user (append randoms if specified)
    print("Splitting into one dataset per user...")
    for j in tqdm(range(n_users)):
        X_user_j = data_file[X_data_name][user_ind_dict[j], :, :]
        n_examples_user_j = X_user_j.shape[0]
        y_user_j = np.ones((n_examples_user_j,))

        N_RANDOM_USERS = n_users//2
        if append_randoms:
            random_users = random.sample(list(range(j)) + list(range(j + 1, n_users)), N_RANDOM_USERS)
            for random_user in random_users:
                example_inds = user_ind_dict[random_user]
                random_example_inds = random.sample(example_inds, n_examples_user_j//N_RANDOM_USERS)
                random_example_inds.sort()

                X_user_j = np.append(X_user_j, data_file[X_data_name][random_example_inds, :, :], axis=0)
                y_user_j = np.append(y_user_j, np.zeros((len(random_example_inds),)), axis=0)

        # Save the data
        X_name_user_j = "X_" + output_name + "_user_" + str(j)
        y_name_user_j = "y_" + output_name + "_user_" + str(j)

        data_file.create_dataset(X_name_user_j, data=X_user_j, dtype=float)
        data_file.create_dataset(y_name_user_j, data=y_user_j, dtype=float)


def parse_args(args):
    """
    Checks that the provided arguments are valid and returns data file name.
    """

    # Check that input args are valid
    assert args.train_frac + args.valid_frac + args.test_frac == 1, "Specified train/valid/train fractions do not sum to 1."

    if args.step_size is None:
        args.step_size = args.example_length - 1

    if not os.path.isdir(args.output_path):
        response = input("Output path does not exist. Create it? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()
        else:
            os.makedirs(args.output_path)

    n_users = len(os.listdir(args.input_path))

    if args.mode == "joint":
        assert args.n_valid_users is not None, "Number of valid users must be specified in mode 'joint'."
        assert args.n_valid_users < n_users, "Number of valid users must be smaller than the total number of users in the input data."
    elif args.mode == "separated":
        assert args.n_valid_users is None, "n_valid_users should not be specified in mode 'separated'."

    if args.mode == "joint":
        data_file_name = str(args.n_valid_users) + "_users_joint_" + str(int(100 * args.train_frac)) + "_" + str(int(100 * args.valid_frac)) + "_" + str(int(100 * args.test_frac)) + ".hdf5"
    elif args.mode == "separated":
        data_file_name = str(n_users) + "_users_separated_" + str(int(100 * args.train_frac)) + "_" + str(int(100 * args.valid_frac)) + "_" + str(int(100 * args.test_frac)) + ".hdf5"
    elif args.mode == "mixed":
        data_file_name = str(n_users) + "_users_mixed_" + str(int(100 * args.train_frac)) + "_" + str(int(100 * args.valid_frac)) + "_" + str(int(100 * args.test_frac)) + ".hdf5"

    if os.path.isfile(args.output_path + data_file_name):
        response = input("Output directory contains identical dataset. Do you want to overwrite it? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()

    return data_file_name


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read preprocessed typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write generated examples to.")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=("joint", "separated", "mixed"), default="mixed",
                        help="Determines if the datasets should be separated per user (separated), put in one joint dataset with only valid users in training set "
                        "(joint) or put in joint dataset with data mixed (mixed). See code comments for details.")
    parser.add_argument("-e", "--example_length", metavar="EXAMPLE_LENGTH", type=int, default=18, help="Number of keystrokes to use as one data example.")
    parser.add_argument("-train", "--train_frac", metavar="TRAIN_FRAC", type=float, default=0.8, help="Fraction of examples to use as training data.")
    parser.add_argument("-valid", "--valid_frac", metavar="VALID_FRAC", type=float, default=0.1, help="Fraction of examples to use as validation data.")
    parser.add_argument("-test", "--test_frac", metavar="TEST_FRAC", type=float, default=0.1, help="Fraction of examples to use as test data.")
    parser.add_argument("-n_valid", "--n_valid_users", metavar="N_VALID_USERS", type=int, default=None, help="Number of random users to select as authorized users.")
    parser.add_argument("-s", "--step_size", metavar="STEP_SIZE", type=int, default=None,
                        help="Step size to use when generating additional examples. A step size s will yield (example_length - 1)//s additional examples per original example.")
    args = parser.parse_args()

    # Check that arguments are valid and determine data file name
    data_file_name = parse_args(args)
    n_users = len(os.listdir(args.input_path))

    # Create h5py file to store the data in
    data_file = h5py.File(args.output_path + data_file_name, "w")

    # Create the regular examples
    X, y = create_examples(args.input_path, data_file, args.example_length)

    if args.mode == "joint":

        # Split into set of valid (v) and unknown (u) users (and relabel accordingly)
        X_v, y_v, X_u, y_u = utils.split_on_users(X, y, n_valid_users=args.n_valid_users, pick_random=False)

        # Split the data into train/valid/test
        X_train, y_train, X_valid, y_valid, X_test_v, y_test_v = utils.split_per_user(X_v, y_v, train_frac=args.train_frac, valid_frac=args.valid_frac,
                                                                                     test_frac=args.test_frac, shuffle=False)

        # Save data from unknowns to be used as test data
        X_test_u, y_test_u = X_u, y_u
        data_file.create_dataset("X_test_unknown", data=X_test_u, dtype=float)
        data_file.create_dataset("y_test_unknown", data=y_test_u, dtype=float)

        # Generate additional examples for each set (except the unknowns) and save
        data_file.create_dataset("X_train", data=X_train, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_train", data=y_train, maxshape=(None, n_users), dtype=float)
        generate_examples_from_adjacents(X_train, y_train, "train", data_file, args.step_size)

        data_file.create_dataset("X_valid", data=X_valid, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_valid", data=y_valid, maxshape=(None, n_users), dtype=float)
        generate_examples_from_adjacents(X_valid, y_valid, "valid", data_file, args.step_size)

        data_file.create_dataset("X_test_valid", data=X_test_v, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_test_valid", data=y_test_v, maxshape=(None, n_users), dtype=float)
        generate_examples_from_adjacents(X_test_v, y_test_v, "test_valid", data_file, args.step_size)

    elif args.mode == "separated":

        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_per_user(X, y, args.train_frac, args.valid_frac, args.test_frac, shuffle=False)

        # Generate additional examples and save in the h5py file
        data_file.create_dataset("X_train_full", data=X_train, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_train_full", data=y_train, maxshape=(None, n_users), dtype=float)
        generate_examples_from_adjacents(X_train, y_train, "train_full", data_file, args.step_size)

        data_file.create_dataset("X_valid_full", data=X_valid, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_valid_full", data=y_valid, maxshape=(None, n_users), dtype=float)
        generate_examples_from_adjacents(X_valid, y_valid, "valid_full", data_file, args.step_size)

        data_file.create_dataset("X_test_full", data=X_test, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_test_full", data=y_test, maxshape=(None, n_users), dtype=float)
        generate_examples_from_adjacents(X_test, y_test, "test_full", data_file, args.step_size)

        # Split the data so that every user gets its own dataset (with some random data from other users appended)
        split_all_users("X_train_full", "y_train_full", "train", data_file, append_randoms=True)
        split_all_users("X_valid_full", "y_valid_full", "valid", data_file, append_randoms=True)
        split_all_users("X_test_full", "y_test_full", "test", data_file, append_randoms=True)

    elif args.mode == "mixed":

        args.loss_thresh = None

        # Split the data into train/valid/test
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_per_user(X, y, train_frac=args.train_frac, valid_frac=args.valid_frac,
                                                                                  test_frac=args.test_frac, shuffle=True)

        data_file.create_dataset("X_train", data=X_train, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_train", data=y_train, maxshape=(None, n_users), dtype=float)

        data_file.create_dataset("X_valid", data=X_valid, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_valid", data=y_valid, maxshape=(None, n_users), dtype=float)

        data_file.create_dataset("X_test", data=X_test, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_test", data=y_test, maxshape=(None, n_users), dtype=float)

        # Create datasets for triplet training data
        data_file.create_dataset("X_train_anchors", shape=(0, args.example_length, FEATURE_LENGTH),
                                 maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_train_anchors", shape=(0, n_users), maxshape=(None, n_users), dtype=float)

        data_file.create_dataset("X_train_positives", shape=(0, args.example_length, FEATURE_LENGTH),
                                 maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_train_positives", shape=(0, n_users), maxshape=(None, n_users), dtype=float)

        data_file.create_dataset("X_train_negatives", shape=(0, args.example_length, FEATURE_LENGTH),
                                 maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_train_negatives", shape=(0, n_users), maxshape=(None, n_users), dtype=float)

        # Generate training triplets
        generate_triplets.create_triplets(args, "X_train", "y_train", output_name="train", n_examples_per_anchor=10, data_file=data_file)

        # Create datasets for triplet validation data
        data_file.create_dataset("X_valid_anchors", shape=(0, args.example_length, FEATURE_LENGTH),
                                 maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_valid_anchors", shape=(0, n_users), maxshape=(None, n_users), dtype=float)

        data_file.create_dataset("X_valid_positives", shape=(0, args.example_length, FEATURE_LENGTH),
                                 maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_valid_positives", shape=(0, n_users), maxshape=(None, n_users), dtype=float)

        data_file.create_dataset("X_valid_negatives", shape=(0, args.example_length, FEATURE_LENGTH),
                                 maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
        data_file.create_dataset("y_valid_negatives", shape=(0, n_users), maxshape=(None, n_users), dtype=float)

        # Generate validation triplets
        generate_triplets.create_triplets(args, "X_valid", "y_valid", output_name="valid", n_examples_per_anchor=100, data_file=data_file)

    print("\nExample generation successful!")
    print("Datasets are saved in: {}".format(args.output_path + data_file_name))
    print("Use h5py to read them.")


if __name__ == "__main__":
    main()
