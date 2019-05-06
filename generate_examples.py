import os
import argparse

import numpy as np
from tqdm import tqdm

import util


def create_examples(input_path, example_length):
    """
    Loads all digraph data in input_path, creates examples of length example_length
    and corresponding labels.

    Returns:
    Matrix X of shape (#examples, example_length, feature_length)
    Matrix y of shape (#examples, #users)
    """

    X = []
    y = []

    n_users = len(os.listdir(input_path))

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
                    X.append(example)
                    y.append(i)
                    example = []

    X = np.asarray(X)
    y = np.asarray(y)

    y = util.index_to_one_hot(y, n_users)

    return X, y, n_users


def generate_examples_from_adjacents(X, y, example_length, step_size=1):
    """
    Parses through the data and generates (example_length - 1)//step_size
    additional examples from every pair of adjacent examples with the same label.
    The default step=1 generates *all* possible additional examples.

    The generated examples are appended to the original data, and the full dataset is returned.
    """

    assert step_size <= example_length - 1 and step_size > 0, "Invalid step size. Must have 0 < step <= example_length - 1"

    n_examples = X.shape[0]

    X_additional = np.empty((0,) + X.shape[1:])
    y_additional = np.empty((0, y.shape[1]))

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

    X = np.vstack((X, X_additional))
    y = np.vstack((y, y_additional))

    return X, y


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read preprocessed typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write augmented data to.")
    parser.add_argument("-e", "--example_length", metavar="EXAMPLE_LENGTH", type=int, default=18, help="Number of keystrokes to use as one data example.")
    parser.add_argument("-train", "--train_frac", metavar="TRAIN_FRAC", type=float, default=0.8, help="Fraction of examples to use as training data.")
    parser.add_argument("-valid", "--valid_frac", metavar="VALID_FRAC", type=float, default=0.1, help="Fraction of examples to use as validation data.")
    parser.add_argument("-test", "--test_frac", metavar="TEST_FRAC", type=float, default=0.1, help="Fraction of examples to use as test data.")
    parser.add_argument("-n_valid", "--n_valid_users", metavar="N_VALID_USERS", type=int, default=8, help="Number of random users to select as authorixed users.")
    parser.add_argument("-s", "--step_size", metavar="STEP_SIZE", type=int, default=None,
                        help="Step size to use when generating additional examples. A step size s will yield (example_length - 1)//s additional examples per original example.")
    args = parser.parse_args()

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

    if len(os.listdir(args.output_path)) > 0:
        response = input("Output directory is not empty. Numpy files may be overwritten. Continue? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()

    # Create the regular examples
    X, y, n_users = create_examples(args.input_path, args.example_length)

    # Split into set of valid (v) and unknown (u) users (and relabel accordingly)
    assert args.n_valid_users < n_users, "Number of valid users must be smaller than the total number of users in the input data."
    X_v, y_v, X_u, y_u = util.split_on_users(X, y, n_valid_users=args.n_valid_users, pick_random=False)

    # Split the data into train/valid/test
    X_train, y_train, X_valid, y_valid, X_test_v, y_test_v = util.split_per_user(X_v, y_v, train_frac=args.train_frac, valid_frac=args.valid_frac,
                                                                                 test_frac=args.test_frac, shuffle=False)
    X_test_u, y_test_u = X_u, y_u

    # Concatenate test data from valid/unknown users
    X_test = np.vstack((X_test_v, X_test_u))
    y_test = np.vstack((y_test_v, y_test_u))

    # Generate additional examples for each set
    X_train, y_train = generate_examples_from_adjacents(X_train, y_train, args.example_length, args.step_size)
    X_valid, y_valid = generate_examples_from_adjacents(X_valid, y_valid, args.example_length, args.step_size)
    X_test, y_test = generate_examples_from_adjacents(X_test, y_test, args.example_length, args.step_size)

    # Save the data in output_path
    np.save(args.output_path + "X_train.npy", X_train)
    np.save(args.output_path + "y_train.npy", y_train)
    np.save(args.output_path + "X_valid.npy", X_valid)
    np.save(args.output_path + "y_valid.npy", y_valid)
    np.save(args.output_path + "X_test.npy", X_test)
    np.save(args.output_path + "y_test.npy", y_test)

    print("\nExample generation successful!")
    print("Numpy binary files were saved in: {}".format(args.output_path))
    print("Use np.load(file) to read them.")


if __name__ == "__main__":
    main()
