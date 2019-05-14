"""
Builds a siamese CNN and trains it to embed typing data from same user to be similar,
and typing data from different users to be dissimilar.

Adapted from:
https://github.com/divyashan/time_series/blob/master/models/supervised/siamese_triplet_keras.py
"""

import os
import argparse

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import keras.backend as K

# import keras
# from keras.layers import Sequential
# from keras.layers import Activation
# from keras.layers import SimpleRNN
# from keras import initializers

import utils

# Parameters
ALPHA = 1  # Triplet loss threshold
LEARNING_RATE = 1e-6
EPOCHS = 100
BATCH_SIZE = 50


def build_tower_cnn_model(input_shape):
    """
    Builds a CNN-model for embedding single examples of data.
    """

    x0 = Input(input_shape, name='Input')

    kernel = 5
    n_channels = [16]
    x = x0
    for i in range(len(n_channels)):
        x = Conv1D(n_channels[i], kernel_size=kernel, strides=2, activation='relu', padding='same')(x)
        x = MaxPooling1D(5)(x)

    x = Flatten()(x)
    y = Dense(40, name='dense_encoding')(x)

    model = Model(inputs=x0, outputs=y)

    return model


def _euclidean_distance(vects):
    """
    Computes euclidean distance between tuple of vectors.
    """
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def _eucl_dist_output_shape(shapes):
    """
    Wat?
    """
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def _triplet_distance(vects):
    """
    Computes triplet loss for single triplet.
    """
    A, P, N = vects
    return _euclidean_distance([A, P]) - _euclidean_distance([A, N]) + ALPHA


def build_triplet_model(input_shape, tower_model):
    """
    Builds a model that takes a triplet as input, feeds them through
    tower_model and outputs the triplet loss of the embeddings.
    """
    input_A = Input(input_shape)
    input_B = Input(input_shape)
    input_C = Input(input_shape)

    tower_model.summary()

    x_A = tower_model(input_A)
    x_B = tower_model(input_B)
    x_C = tower_model(input_C)

    distance = Lambda(_triplet_distance, output_shape=_eucl_dist_output_shape)([x_A, x_B, x_C])

    model = Model([input_A, input_B, input_C], distance, name='siamese')

    return model


def plot_with_PCA(X_embedded, y):
    """
    Applies PCA (with n_components=2) to X_embedded and plots
    resulting (x_1, x_2) in 2D, with color indicating class.

    Scikit-learn has PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    pass  # TODO


def plot_with_t_SNE(X_embedded, y):
    """
    Applies t-SNE (with n_components=2) to X_embedded and plots
    resulting (x_1, x_2) in 2D, with color indicating class.

    Scikit-learn has t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    pass  # TODO


def parse_args(args):
    """
    Checks that input args are valid.
    """

    if args.save_path is not None:
        if not os.path.isdir(args.save_path):
            response = input("Save path does not exist. Create it? (Y/n) >> ")
            if response.lower() not in ["y", "yes", "1", ""]:
                exit()
            else:
                os.makedirs(args.save_path)

    if args.metrics_path is not None:
        if not os.path.isdir(args.metrics_path):
            response = input("Metrics path does not exist. Create it? (Y/n) >> ")
            if response.lower() not in ["y", "yes", "1", ""]:
                exit()
            else:
                os.makedirs(args.metrics_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="data_path", metavar="DATA_PATH", help="Path to read examples from.")
    parser.add_argument("-s", "--save_path", metavar="SAVE_PATH", default=None, help="Path to save trained model to. If no path is specified checkpoints are not saved.")
    parser.add_argument("-m", "--metrics-path", metavar="METRICS_PATH", default=None, help="Path to save additional performance metrics to (for debugging purposes).")
    args = parser.parse_args()

    # Load training triplets and validation triplets
    X_train_anchors, _ = utils.load_examples(args.data_path, "train_anchors")
    X_train_positives, _ = utils.load_examples(args.data_path, "train_positives")
    X_train_negatives, _ = utils.load_examples(args.data_path, "train_negatives")
    X_valid_anchors, _ = utils.load_examples(args.data_path, "valid_anchors")
    X_valid_positives, _ = utils.load_examples(args.data_path, "valid_positives")
    X_valid_negatives, _ = utils.load_examples(args.data_path, "valid_negatives")

    # Build model
    input_shape = X_train_anchors.shape[1:]
    tower_model = build_tower_cnn_model(input_shape)  # single input model
    triplet_model = build_triplet_model(input_shape, tower_model)  # siamese model

    # Compile model
    adam = Adam(lr=LEARNING_RATE)
    triplet_model.compile(optimizer=adam, loss='mean_squared_error')

    # Create dummy y = 0 (since output of siamese model is triplet loss)
    y_train_dummy = np.zeros((X_train_anchors.shape[0],))  # dummy y for triplet training
    y_valid_dummy = np.zeros((X_valid_anchors.shape[0],))

    # Train the model
    triplet_model.fit([X_train_anchors, X_train_positives, X_train_negatives], y_train_dummy,
                      validation_data=([X_valid_anchors, X_valid_positives, X_valid_negatives], y_valid_dummy),
                      epochs=EPOCHS, batch_size=BATCH_SIZE)


    # This is how you can embed data using the trained model:

    # embedding = tower_model.predict(X)


    # Other things that may be useful later:

    # Evaluation function found in util.py of the Jiffy git repo
    # print(evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test))

    # Training using fit_generator (probably not necessary for us because data is limited)
    # triplet_model.fit_generator(gen_batch(X_train, tr_trip_idxs, batch_size, dummy_y), epochs=1, steps_per_epoch=n_batches_per_epoch)


if __name__ == "__main__":
    main()
