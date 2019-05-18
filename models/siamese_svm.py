import os
import argparse
import numpy as np
from sklearn import svm
from keras.models import load_model
import utils
import cnn_keras_siamese
from keras.utils import CustomObjectScope


def generateExamplesFromTriplets(model, anchors, positives, negatives):
    n_examples = anchors.shape[0]
    X = np.zeros((2*n_examples, 40))
    X[:n_examples,:] = getDiffEncodings(model, anchors, positives)
    X[n_examples:,:]= getDiffEncodings(model, anchors, negatives)
    y = np.zeros((n_examples*2,))
    y[:n_examples] = 1
    return X, y


def getDiffEncodings(model, Z1, Z2):
    return np.absolute(model.predict(Z1)-model.predict(Z2))
    

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="triplets_path", metavar="TRIPLETS_PATH", help="Path to read triplets from.")
    parser.add_argument(dest="model_path", metavar="MODEL_PATH", help="Path to read model from.")
    args = parser.parse_args()

    #Load training triplets and validation triplets
    X_train_anchors, _ = utils.load_examples(args.triplets_path, "train_anchors")
    X_train_positives, _ = utils.load_examples(args.triplets_path, "train_positives")
    X_train_negatives, _ = utils.load_examples(args.triplets_path, "train_negatives")
    X_valid_anchors, _ = utils.load_examples(args.triplets_path, "valid_anchors")
    X_valid_positives, _ = utils.load_examples(args.triplets_path, "valid_positives")
    X_valid_negatives, _ = utils.load_examples(args.triplets_path, "valid_negatives")

    # Load model
    with CustomObjectScope({'_euclidean_distance': cnn_keras_siamese._euclidean_distance,
        'ALPHA': cnn_keras_siamese.ALPHA}):
        model = load_model(args.model_path)
        model.compile(optimizer='adam', loss='mean_squared_error')

    X_train, y_train = generateExamplesFromTriplets(model, X_train_anchors, X_train_positives, X_train_negatives)
    X_valid, y_valid = generateExamplesFromTriplets(model, X_valid_anchors, X_valid_positives, X_valid_negatives)

    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm,:]
    y_train = y_train[perm]

    perm = np.random.permutation(X_valid.shape[0])
    X_valid = X_valid[perm,:]
    y_valid = y_valid[perm]

    clf = svm.SVC(gamma='scale', verbose=True)
    clf.fit(X_train[:10000,:], y_train[:10000])

    print("\n\nValidation error: " + str(np.sum(np.absolute(clf.predict(X_valid[:10000,:]) - y_valid[:10000]))/float(10000)))


if __name__ == "__main__":
    main()