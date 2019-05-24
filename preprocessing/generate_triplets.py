import os
import sys
import argparse

import numpy as np
from tqdm import tqdm
import h5py

assert os.getcwd().split("/")[-1] == "typing-net", "Preprocessing scripts must run from typing-net/ directory."
sys.path.insert(0, 'models/')  # so that utils can be imported
import utils


# Constants
FEATURE_LENGTH = 6


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
    i = 0
    for user_file_name in tqdm(os.listdir(input_path)):
        if user_file_name[0] == ".":
            n_users -= 1
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
        i += 1

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


def create_triplets(X_data_name, y_data_name, output_name, data_file):
    """
    Takes a dataset X, y stored in a hdf5 file and generates one triplet (A, P, N)
    for every example in X.

    Each example in X is Anchor in one triplet. The Positive of each triplet is selected as a random
    different example from the same user as Anchor. The Negative of each triplet is selected as a random
    different example from a different user than Anchor.
    """
    WRITE_CHUNK_SIZE = 1000

    n_examples = data_file[X_data_name].shape[0]
    example_length = data_file[X_data_name].shape[1]
    n_users = data_file[y_data_name].shape[1]

    X_anchors = []
    y_anchors = []
    X_positives = []
    y_positives = []
    X_negatives = []
    y_negatives = []

    X_anchors_name = "X_" + output_name + "_anchors"
    y_anchors_name = "y_" + output_name + "_anchors"
    X_positives_name = "X_" + output_name + "_positives"
    y_positives_name = "y_" + output_name + "_positives"
    X_negatives_name = "X_" + output_name + "_negatives"
    y_negatives_name = "y_" + output_name + "_negatives"

    # Generate dictionary mapping user id to example indices
    print("Preparing generation of triplets...")
    user_ind_dict = {j: [] for j in range(n_users)}
    for i in tqdm(range(n_examples)):
        j = utils.one_hot_to_index(data_file[y_data_name][i, :])
        user_ind_dict[j].append(i)

    print("Creating triplets from single examples...")
    for i in tqdm(range(n_examples)):

        anchor_X = np.expand_dims(data_file[X_data_name][i, :, :], axis=0)
        anchor_y = np.expand_dims(data_file[y_data_name][i, :], axis=0)

        positives_inds = user_ind_dict[utils.one_hot_to_index(anchor_y[0])]

        for ii in range(500):
            positive_choice = np.random.choice(positives_inds)
            positive_X = np.expand_dims(data_file[X_data_name][positive_choice, :, :], axis=0)
            positive_y = np.expand_dims(data_file[y_data_name][positive_choice, :, ], axis=0)

            user_ids = list(user_ind_dict.keys())
            user_ids.remove(utils.one_hot_to_index(anchor_y[0]))
            random_user = np.random.choice(user_ids)
            negatives_inds = user_ind_dict[random_user]
            negative_choice = np.random.choice(negatives_inds)
            negative_X = np.expand_dims(data_file[X_data_name][negative_choice, :, :], axis=0)
            negative_y = np.expand_dims(data_file[y_data_name][negative_choice, :], axis=0)

            X_anchors.extend(anchor_X)
            y_anchors.extend(anchor_y)
            X_positives.extend(positive_X)
            y_positives.extend(positive_y)
            X_negatives.extend(negative_X)
            y_negatives.extend(negative_y)

        if len(X_anchors) >= WRITE_CHUNK_SIZE or i == n_examples-1:

            data_file[X_anchors_name].resize(data_file[X_anchors_name].shape[0] + len(X_anchors), axis=0)
            data_file[X_anchors_name][-len(X_anchors):] = np.asarray(X_anchors)
            data_file[y_anchors_name].resize(data_file[y_anchors_name].shape[0] + len(y_anchors), axis=0)
            data_file[y_anchors_name][-len(y_anchors):] = np.asarray(y_anchors)

            data_file[X_positives_name].resize(data_file[X_positives_name].shape[0] + len(X_positives), axis=0)
            data_file[X_positives_name][-len(X_positives):] = np.asarray(X_positives)
            data_file[y_positives_name].resize(data_file[y_positives_name].shape[0] + len(y_positives), axis=0)
            data_file[y_positives_name][-len(y_positives):] = np.asarray(y_positives)

            data_file[X_negatives_name].resize(data_file[X_negatives_name].shape[0] + len(X_negatives), axis=0)
            data_file[X_negatives_name][-len(X_negatives):] = np.asarray(X_negatives)
            data_file[y_negatives_name].resize(data_file[y_negatives_name].shape[0] + len(y_negatives), axis=0)
            data_file[y_negatives_name][-len(y_negatives):] = np.asarray(y_negatives)

            X_anchors = []
            y_anchors = []
            X_positives = []
            y_positives = []
            X_negatives = []
            y_negatives = []


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read preprocessed typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write generated examples to.")
    parser.add_argument("-e", "--example_length", metavar="EXAMPLE_LENGTH", type=int, default=18, help="Number of keystrokes to use as one data example.")
    parser.add_argument("-train", "--train_frac", metavar="TRAIN_FRAC", type=float, default=0.8, help="Fraction of examples to use to generate triplets to be used as training data.")
    parser.add_argument("-valid", "--valid_frac", metavar="VALID_FRAC", type=float, default=0.1, help="Fraction of examples to use to generate triplets to be used as validation data.")
    parser.add_argument("-test", "--test_frac", metavar="TEST_FRAC", type=float, default=0.1, help="Fraction of examples to use as test data (left as singles).")
    parser.add_argument("-s", "--step_size", metavar="STEP_SIZE", type=int, default=None,
                        help="Step size to use when generating additional examples. A step size s will yield (example_length - 1)//s additional examples per original example.")
    args = parser.parse_args()

    # Check that input args are valid
    assert args.train_frac + args.valid_frac + args.test_frac == 1, "Specified train/valid/train fractions do not sum to 1."

    if args.step_size is None:
        args.step_size = 1

    if not os.path.isdir(args.output_path):
        response = input("Output path does not exist. Create it? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()
        else:
            os.makedirs(args.output_path)

    data_file_name = "triplets_" + str(int(100*args.train_frac)) + "_" + str(int(100*args.valid_frac)) + "_" + str(int(100*args.test_frac)) + ".hdf5"

    if os.path.isfile(args.output_path + data_file_name):
        response = input("Output directory contains identical dataset. Do you want to overwrite it? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()

    # Create h5py file to store the data in
    data_file = h5py.File(args.output_path + data_file_name, "w")

    # Create the regular examples
    X, y = create_examples(args.input_path, data_file, args.example_length)
    n_users = y.shape[1]

    # Split the data into train/valid/test
    X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_per_user(X, y, train_frac=args.train_frac, valid_frac=args.valid_frac, test_frac=args.test_frac, shuffle=False)

    # Generate additional examples for each set and save
    data_file.create_dataset("X_train_singles", data=X_train, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
    data_file.create_dataset("y_train_singles", data=y_train, maxshape=(None, n_users), dtype=float)
    # generate_examples_from_adjacents(X_train, y_train, "train_singles", data_file, args.step_size)

    data_file.create_dataset("X_valid_singles", data=X_valid, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
    data_file.create_dataset("y_valid_singles", data=y_valid, maxshape=(None, n_users), dtype=float)
    # generate_examples_from_adjacents(X_valid, y_valid, "valid_singles", data_file, args.step_size)

    data_file.create_dataset("X_test", data=X_test, maxshape=(None, args.example_length, FEATURE_LENGTH), dtype=float)
    data_file.create_dataset("y_test", data=y_test, maxshape=(None, n_users), dtype=float)
    # generate_examples_from_adjacents(X_test, y_test, "test", data_file, args.step_size)

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
    create_triplets("X_train_singles", "y_train_singles", output_name="train", data_file=data_file)

    # Delete the train "singles"
    del data_file["X_train_singles"]
    del data_file["y_train_singles"]

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
    create_triplets("X_valid_singles", "y_valid_singles", output_name="valid", data_file=data_file)

    # Delete the validation "singles"
    del data_file["X_valid_singles"]
    del data_file["y_valid_singles"]

    print("\nTriplet generation successful!")
    print("Datasets are saved in: {}".format(args.output_path + data_file_name))
    print("Use h5py to read them.")


if __name__ == "__main__":
    main()
