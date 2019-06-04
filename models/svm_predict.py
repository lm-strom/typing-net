import os
import argparse
import random

import numpy as np
from sklearn import svm
from keras.models import load_model, Model
from keras.layers import Input, Lambda
from keras.utils import CustomObjectScope
import keras.backend as K

import utils
import cnn_siamese_online


def build_pair_distance_model(tower_model, input_shape):
    """
    Builds a model that takes a triplet as input and returns
    abs(A - P) and abs(A - N)
    """

    input_anchor = Input(input_shape)
    input_positive = Input(input_shape)
    input_negative = Input(input_shape)

    embedd_anchor = tower_model(input_anchor)
    embedd_positive = tower_model(input_positive)
    embedd_negative = tower_model(input_negative)

    tower_output_shape = tower_model.layers[-1].output_shape

    abs_difference = Lambda(lambda z: K.abs(z[0] - z[1]), output_shape=tower_output_shape)

    positive_pair_dist = abs_difference([embedd_anchor, embedd_positive])
    negative_pair_dist = abs_difference([embedd_anchor, embedd_negative])

    pair_distance_model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=[positive_pair_dist, negative_pair_dist])

    return pair_distance_model


def shuffle(X, y):

    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    y = y[perm]

    return X, y


def accuracy_FAR_FRR(y_true, y_pred):

    n_examples = y_true.shape[0]

    correct = 0
    FAR_errors = 0
    FRR_errors = 0
    for i in range(n_examples):

        if y_true[i] == y_pred[i]:
            correct += 1

        elif y_true[i] == 0 and y_pred[i] == 1:
            FAR_errors += 1

        elif y_true[i] == 1 and y_pred[i] == 0:
            FRR_errors += 1

    accuracy = float(correct) / n_examples
    FAR = float(FAR_errors) / (n_examples - np.sum(y_true))
    FRR = float(FRR_errors) / np.sum(y_true)

    return accuracy, FAR, FRR


def ensemble_accuracy_FAR_FRR(pair_distance_model, svm_model, X_test_separated, user, ensemble_size):
    """
    Compute ensemble accuracy, FAR and FRR
    """
    n_FA, n_FR, n_correct = 0, 0, 0
    n_trials = 0

    n_users = len(X_test_separated)
    X_test_user = X_test_separated[user]
    n_examples_user = X_test_user.shape[0]

    for a in range(n_examples_user):

        anchor = np.expand_dims(X_test_user[a, :, :], 0)

        # Select positives and negatives
        positives = []
        negatives = []
        other_user = random.choice(list(range(user)) + list(range(user + 1, n_users)))  # pick a random different user
        candidate_positives = list(range(a)) + list(range(a + 1, n_examples_user))
        candidate_negatives = list(range(X_test_separated[other_user].shape[0]))

        for i in range(ensemble_size):
            p = random.choice(candidate_positives)  # pick a random example from the same user
            n = random.choice(candidate_negatives)  # pick a random example from that user
            candidate_positives.remove(p)
            candidate_negatives.remove(n)

            positive = np.expand_dims(X_test_user[p, :, :], 0)
            positives.append(positive)
            negative = np.expand_dims(X_test_separated[other_user][n, :, :], 0)
            negatives.append(negative)

        # Reformat data for prediction
        anchors = np.tile(anchor.T, ensemble_size).T
        positives = np.squeeze(np.array(positives), axis=1)
        negatives = np.squeeze(np.array(negatives), axis=1)


        # Predict
        AP_dists, AN_dists = pair_distance_model.predict([anchors, positives, negatives])
        y_pos_preds = svm_model.predict(AP_dists)
        y_neg_preds = svm_model.predict(AN_dists)
        y_pos_pred = (np.sum(y_pos_preds) > (ensemble_size // 2))
        y_neg_pred = (np.sum(y_neg_preds) > (ensemble_size // 2))

        # Evaluate
        if y_pos_pred == 1:
            n_correct += 1
        else:
            n_FR += 1

        if y_neg_pred == 0:
            n_correct += 1
        else:
            n_FA += 1

        n_trials += 1

    return n_FA, n_FR, n_correct, n_trials


def predict_and_evaluate(pair_distance_model, svm_model, X_test_separated, ensemble_size):
    """
    Make predictions on X_test_separated (list of data per user)
    using pair_distance_model and svm_model, and ensembling of size ensemble_size.
    """
    n_users = len(X_test_separated)

    # Evaluate
    n_FA_tot, n_FR_tot, n_correct_tot = 0, 0, 0
    n_trials_tot = 0
    for user in range(n_users):
        n_FA, n_FR, n_correct, n_trials = ensemble_accuracy_FAR_FRR(pair_distance_model, svm_model, X_test_separated, user, ensemble_size)
        n_FA_tot += n_FA
        n_FR_tot += n_FR
        n_correct_tot += n_correct
        n_trials_tot += n_trials

    accuracy = float(n_correct_tot) / (2 * n_trials_tot)
    FAR = float(n_FA_tot) / n_trials_tot
    FRR = float(n_FR_tot) / n_trials_tot

    return accuracy, FAR, FRR


def parse_args(args):
    """
    Checks that input args are valid.
    """

    assert os.path.isfile(args.triplets_path), "The specified triplet file does not exist."
    assert os.path.isfile(args.model_path), "The specified model file does not exist."

    if args.read_batches is not False:
        if args.read_batches.lower() in ("y", "yes", "1", "", "true", "t"):
            args.read_batches = True
        else:
            args.read_batches = False

    args.ensemble = int(args.ensemble)
    assert args.ensemble <= 100, "Invalid ensemble value. Cannot have an ensemble > 100."


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="triplets_path", metavar="TRIPLETS_PATH", help="Path to read triplets from.")
    parser.add_argument(dest="model_path", metavar="MODEL_PATH", help="Path to read model from.")
    parser.add_argument("-e", "--ensemble", metavar="ENSEMBLE", default=1, help="How many examples to ensemble when predicting. Default: 1")
    parser.add_argument("-b", "--read_batches", metavar="READ_BATCHES", default=False, help="If true, data is read incrementally in batches during training.")
    args = parser.parse_args()
    parse_args(args)

    # Load model
    with CustomObjectScope({'_euclidean_distance': cnn_siamese_online._euclidean_distance,
                            'ALPHA': cnn_siamese_online.ALPHA, "relu_clipped": cnn_siamese_online.relu_clipped}):
        tower_model = load_model(args.model_path)
        tower_model.compile(optimizer='adam', loss='mean_squared_error')  # Model was previously not compiled

    X_shape, y_shape = utils.get_shapes(args.triplets_path, "train_anchors")

    # Build model to compute [A, P, N] => [abs(emb(A) - emb(P)), abs(emb(A) - emb(N))]
    pair_distance_model = build_pair_distance_model(tower_model, X_shape[1:])
    pair_distance_model.compile(optimizer="adam", loss="mean_squared_error")  # Need to compile in order to predict

    if not args.read_batches:  # Read all data at once

        # Load training triplets and validation triplets
        X_train_anchors, _ = utils.load_examples(args.triplets_path, "train_anchors")
        X_train_positives, _ = utils.load_examples(args.triplets_path, "train_positives")
        X_train_negatives, _ = utils.load_examples(args.triplets_path, "train_negatives")

        # Get abs(distance) of embeddings
        X_train_1, X_train_0 = pair_distance_model.predict([X_train_anchors, X_train_positives, X_train_negatives])

    # Stack positive and negative examples
    X_train = np.vstack((X_train_1, X_train_0))
    y_train = np.hstack((np.ones(X_train_1.shape[0], ), np.zeros(X_train_0.shape[0],)))

    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train)

    # Train SVM
    svm_model = svm.SVC(gamma='scale', verbose=True)
    svm_model.fit(X_train[:20000, :], y_train[:20000])

    # Load test data
    _, y_test_shape = utils.get_shapes(args.triplets_path, "test")
    n_users = y_test_shape[1]
    X_test_separated = []
    for j in range(n_users):
        X_test_j = utils.load_X(args.triplets_path, "test_" + str(j))
        X_test_separated.append(X_test_j)

    # Predict and evaluate
    accuracy, FAR, FRR = predict_and_evaluate(pair_distance_model, svm_model, X_test_separated, args.ensemble)

    print("\n---- Test Results ----")
    print("Accuracy = {}".format(accuracy))
    print("FAR = {}".format(FAR))
    print("FRR = {}".format(FRR))


if __name__ == "__main__":
    main()
