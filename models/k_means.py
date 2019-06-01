import os
import argparse

import numpy as np
from keras.utils import CustomObjectScope
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cnn_siamese_online
import utils

# Parameters
K = 3


def shuffle(X, y):

    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    y = y[perm]

    return X, y


def k_means_PCA(k_means_model, X_emb, y, display_k_means=True):
    """
    Plot resulting k-means clustering. If display_k_means is True, colors will indicate
    which cluster the example belongs to, else colors will indicate class.
    """

    pca = PCA(n_components=2)

    X_emb = StandardScaler().fit_transform(X_emb)
    X_emb = pca.fit_transform(X_emb)

    centroids = k_means_model.cluster_centers_
    centroids_pca = pca.transform(centroids)
    n_clusters = centroids_pca.shape[0]

    if display_k_means:

        labels = k_means_model.labels_
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))

        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors[labels])
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c=colors, marker="x", s=25)
        plt.savefig("k_means_colors.png")

    else:

        labels = utils.one_hot_to_index(y)
        class_colors = cm.rainbow(np.linspace(0, 1, y.shape[1]))
        cluster_colors = cm.rainbow(np.linspace(0, 1, n_clusters))

        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=class_colors[labels])
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c=cluster_colors, marker="x", s=25)
        plt.savefig("class_colors.png")


def compute_cluster_class_fractions(k_means_model, y):
    """
    Computes the fraction of examples of each class in each cluster.
    """

    n_classes = y.shape[1]
    class_labels = utils.one_hot_to_index(y)
    cluster_labels = k_means_model.labels_

    class_clustroid_counts = np.zeros((n_classes, K))
    for i in range(len(class_labels)):
        class_clustroid_counts[class_labels[i], cluster_labels[i]] += 1

    class_clustroid_fractions = class_clustroid_counts / np.sum(class_clustroid_counts, axis=1).reshape(n_classes, 1)

    print("\n---- Class Clustroid Distribution ----")
    for i in range(n_classes):
        print("Class {}: {}".format(i, class_clustroid_fractions[i, :]))


def parse_args(args):
    """
    Checks that input args are valid.
    """
    assert os.path.isfile(args.data_path), "The specified data file does not exist."
    assert os.path.isfile(args.model_path), "The specified model file does not exist."

    if args.read_batches is not False:
        if args.read_batches.lower() in ("y", "yes", "1", "", "true", "t"):
            args.read_batches = True
        else:
            args.read_batches = False


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="data_path", metavar="DATA_PATH", help="Path to read data from.")
    parser.add_argument(dest="model_path", metavar="MODEL_PATH", help="Path to read model from.")
    parser.add_argument("-b", "--read_batches", metavar="READ_BATCHES", default=False, help="If true, data is read incrementally in batches during training.")
    args = parser.parse_args()
    parse_args(args)

    # Load model
    with CustomObjectScope({'_euclidean_distance': cnn_siamese_online._euclidean_distance,
                            'ALPHA': cnn_siamese_online.ALPHA, "relu_clipped": cnn_siamese_online.relu_clipped}):
        tower_model = load_model(args.model_path)
        tower_model.compile(optimizer='adam', loss='mean_squared_error')  # Model was previously not compile

    if not args.read_batches:  # Read all data at once

        # Load training triplets and validation triplets
        X_train, y_train = utils.load_examples(args.data_path, "train")
        X_valid, y_valid = utils.load_examples(args.data_path, "valid")

        # Get abs(distance) of embeddings
        X_train_emb = tower_model.predict(X_train)
        X_valid_emb = tower_model.predict(X_valid)

    else:  # Read data in batches
        raise ValueError("Reading in batches is not implemented yet.")

    # Shuffle the data
    X_train_emb, y_train = shuffle(X_train_emb, y_train)
    X_valid_emb, y_valid = shuffle(X_valid_emb, y_valid)

    # Run k-means on training data
    print("Running K-means...")
    k_means_model = KMeans(n_clusters=K, verbose=0)
    k_means_model.fit(X_train_emb)

    # Plot result
    k_means_PCA(k_means_model, X_train_emb, y_train, display_k_means=True)
    k_means_PCA(k_means_model, X_train_emb, y_train, display_k_means=False)

    # Compute percentage of each class in each cluster
    compute_cluster_class_fractions(k_means_model, y_train)


if __name__ == "__main__":
    main()
