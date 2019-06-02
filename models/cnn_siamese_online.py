"""
Builds a siamese CNN and trains it to embed typing data from same user to be similar,
and typing data from different users to be dissimilar.

Adapted from:
https://github.com/divyashan/time_series/blob/master/models/supervised/siamese_triplet_keras.py
"""

import os
import signal
import argparse
import random

import numpy as np
import h5py

import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import keras.backend as K

import utils

# Constants
PERIOD = 10

# Parameters
ALPHA = 0.3  # Triplet loss threshold
LEARNING_RATE = 0.5e-2
EPOCHS = 1000
BATCH_SIZE = 64

# Global variables
stop_flag = False  # Flag to indicate that training was terminated early
training_complete = False  # Flag to indicate that training is complete


class OnlineTripletGenerator(keras.utils.Sequence):
    """
    Generates batches of "batch hard" triplets with the method described here:
    https://omoindrot.github.io/triplet-loss

    Code is adapted from: https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    """

    def __init__(self, data_path, dataset_name, tower_model, batch_size=300, alpha=ALPHA, triplet_mode="batch_all"):
        "Initialization"

        self.tower_model = tower_model
        self.data_file = h5py.File(data_path, "r")
        self.X_name = "X_" + dataset_name
        self.y_name = "y_" + dataset_name

        self.n_examples = self.data_file[self.X_name].shape[0]
        self.example_length = self.data_file[self.X_name].shape[1]
        self.n_features = self.data_file[self.X_name].shape[2]
        self.n_classes = self.data_file[self.y_name].shape[1]

        self.batch_size = batch_size
        self.alpha = alpha

        self.indices = list(range(self.n_examples))

        assert triplet_mode in ["batch_all", "batch_hard", "random"], "Invalid triplet mode. Choose between batch_all and batch_hard."
        self.triplet_mode = triplet_mode

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.n_examples / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"

        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        self.this_batch_size = len(batch_indices)

        # Load the raw examples
        X_boolean_mask = np.zeros((self.n_examples, self.example_length, self.n_features), dtype=bool)
        X_boolean_mask[batch_indices, :, :] = True
        X_batch = self.data_file[self.X_name][X_boolean_mask].reshape((self.this_batch_size, self.example_length, self.n_features))

        y_boolean_mask = np.zeros((self.n_examples, self.n_classes), dtype=bool)
        y_boolean_mask[batch_indices, :] = True
        y_batch = self.data_file[self.y_name][y_boolean_mask].reshape((self.this_batch_size, self.n_classes))
        labels = np.array(utils.one_hot_to_index(y_batch))

        if self.triplet_mode == "batch_hard":
            # Compute the embeddings of this batch
            embeddings = self.tower_model.predict(X_batch)
            # Generate batch hard triplets
            anchor_inds, positive_inds, negative_inds = self._batch_hard_triplets(embeddings, labels)

        elif self.triplet_mode == "batch_all":
            # Compute the embeddings of this batch
            embeddings = self.tower_model.predict(X_batch)
            # Generate batch hard triplets
            anchor_inds, positive_inds, negative_inds = self._batch_all_triplets(embeddings, labels)

        X_anchors = X_batch[anchor_inds, :, :]
        X_positives = X_batch[positive_inds, :, :]
        X_negatives = X_batch[negative_inds, :, :]
        y_dummy = np.zeros((len(anchor_inds),))

        return [X_anchors, X_positives, X_negatives], y_dummy

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        random.shuffle(self.indices)

    def _pairwise_distances(self, embeddings, squared=False):
        """
        Computes a 2D matrix of distances between all embeddings.
        """

        # Pairwise dot product between all embeddings
        dot_products = np.matmul(embeddings, embeddings.T)

        # Squared L2 norm for each embedding
        square_norms = np.diagonal(dot_products)

        # Pairwise distances
        distances = np.expand_dims(square_norms, 0) - 2.0 * dot_products + np.expand_dims(square_norms, 0)

        # Replace any negative distances with zeros
        distances = np.maximum(distances, 0)

        if not squared:
            distances = np.sqrt(distances)

        return distances

    def _anchor_positive_mask(self, labels):
        """
        Returns a mask of shape (batch_size, batch_size)
        where mask[a, p] is True iff. a and p are distinct
        and have the same label.
        """

        # Check if a and p are distinct
        indices_equal = np.eye(labels.shape[0])
        indices_not_equal = np.logical_not(indices_equal)

        # Check if labels[a] == labels[p]
        labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))

        # AND to get mask
        mask = np.logical_and(indices_not_equal, labels_equal)

        return mask

    def _anchor_negative_mask(self, labels):
        """
        Returns a mask of shape (batch_size, batch_size)
        where mask[a, n] is True iff. a and n have different labels.
        """

        # Check if labels[a] == labels[n]
        labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))

        mask = np.logical_not(labels_equal)

        return mask

    def _triplet_mask(self, labels):
        """
        Returns a mask of shape (batch_size, batch_size, batch_size)
        where mask[a, p, n] is True iff. (a, p, n) is a valid triplet.
        """

        # Check that a, p, n are distinct
        indices_equal = np.eye(labels.shape[0])
        indices_not_equal = np.logical_not(indices_equal)
        a_not_equal_p = np.expand_dims(indices_not_equal, 2)
        a_not_equal_n = np.expand_dims(indices_not_equal, 1)
        p_not_equal_n = np.expand_dims(indices_not_equal, 0)

        distinct_indices = np.logical_and(np.logical_and(a_not_equal_p, a_not_equal_n), p_not_equal_n)

        # Check that a and p have the same label, a and n have different label
        label_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
        a_equal_p = np.expand_dims(label_equal, 2)
        a_equal_n = np.expand_dims(label_equal, 1)

        valid_labels = np.logical_and(a_equal_p, np.logical_not(a_equal_n))

        # Combine the masks
        mask = np.logical_and(distinct_indices, valid_labels)

        return mask

    def _batch_hard_triplets(self, embeddings, labels):
        """
        For each anchor, select the hardest positive and hardest negative
        within the batch. Yields batch_size triplets.
        """

        # Indices of anchors
        anchor_inds = range(self.this_batch_size)

        pairwise_dists = self._pairwise_distances(embeddings, squared=True)

        # For each anchor, pick hardest positive
        mask_anchor_positive = self._anchor_positive_mask(labels)

        ind_cols = np.where(np.sum(mask_anchor_positive, axis=0) == 0)
        mask_anchor_positive[0, ind_cols] = 1

        anchor_positive_dists = np.multiply(mask_anchor_positive, pairwise_dists)  # Set 0 where (a, p) invalid
        positive_inds = np.argmax(anchor_positive_dists, axis=1)  # Find hardest positives

        # For each anchor, pick hardest negative
        mask_anchor_negative = self._anchor_negative_mask(labels)

        ind_cols = np.where(np.sum(mask_anchor_negative, axis=0) == 0)
        mask_anchor_negative[0, ind_cols] = 1

        max_dist = np.amax(pairwise_dists, axis=1, keepdims=True)
        anchor_negative_dists = pairwise_dists + max_dist * (1.0 - mask_anchor_negative)  # Add max dist to invalid negatives
        negative_inds = np.argmin(anchor_negative_dists, axis=1)  # Find hardest negatives

        return anchor_inds, positive_inds, negative_inds

    def _batch_all_triplets(self, embeddings, labels):
        """
        Select all semi-hard and hard triplets.
        """

        pairwise_dists = self._pairwise_distances(embeddings, squared=False)

        anchor_positive_dists = np.expand_dims(pairwise_dists, 2)
        anchor_negative_dists = np.expand_dims(pairwise_dists, 1)

        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.alpha

        # Remove invalid triplets
        mask = self._triplet_mask(labels)
        triplet_loss = np.multiply(mask, triplet_loss)

        # Set negative losses to zero
        triplet_loss = np.maximum(triplet_loss, 0.0)

        # Find triplets where loss > 0
        anchor_inds, positive_inds, negative_inds = np.where((triplet_loss > 1e-16))  # & (triplet_loss <= 1))

        # Convert to lists
        anchor_inds = anchor_inds.tolist()
        positive_inds = positive_inds.tolist()
        negative_inds = negative_inds.tolist()

        return anchor_inds, positive_inds, negative_inds


class TerminateOnFlag(Callback):
    """
    Callback that terminates training at the end of an epoch if stop_flag is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        if stop_flag:
            self.model.stop_training = True


def handler(signum, frame):
    """
    Flags stop_flag if CTRL-C is received.
    """
    global training_complete

    if not training_complete:
        print('\nCTRL+C signal received. Training will finish after current batch.')
        global stop_flag
        stop_flag = True
    else:
        exit()


def setup_callbacks(save_path):
    """
    Sets up callbacks for early stopping and model saving.
    """

    signal.signal(signal.SIGINT, handler)

    callback_list = []

    callback_list.append(TerminateOnFlag())  # Terminate training if CTRL+C

    if save_path is not None:
        model_checkpoint = ModelCheckpoint(save_path + "_class_model_{epoch:02d}_{val_loss:.2f}.hdf5", monitor="val_loss", save_best_only=True, verbose=1, period=10)  # Save model every 10 epochs
        callback_list.append(model_checkpoint)

    return callback_list


def _euclidean_distance(vects):
    """
    Computes euclidean distance between tuple of vectors.
    """
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def _cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def _cos_dist_output_shape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def _eucl_dist_output_shape(shapes):
    """
    Returns output shape of euclidean distance computation.
    """
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def _triplet_distance(vects):
    """
    Computes triplet loss for single triplet.
    """

    A, P, N = vects
    return K.maximum(_euclidean_distance([A, P]) - _euclidean_distance([A, N]) + ALPHA, 0.0)

def relu_clipped():
    pass


def build_tower_cnn_model(input_shape):
    """
    Builds a CNN-model for embedding single examples of data.
    """

    x0 = Input(input_shape, name='Input')

    kernel = 5
    n_channels = [24]
    x = x0
    for i in range(len(n_channels)):
        x = Conv1D(n_channels[i], kernel_size=kernel, strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(5)(x)

    x = Flatten()(x)
    y = Dense(40, name='dense_encoding')(x)
    y = Lambda(lambda  x: K.l2_normalize(x,axis=1))(y)

    model = Model(inputs=x0, outputs=y)

    return model


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


def parse_args(args):
    """
    Checks that input args are valid.
    """

    if args.save_weights_path is not None:
        if not os.path.isdir(args.save_weights_path):
            response = input("Save weights path does not exist. Create it? (Y/n) >> ")
            if response.lower() not in ["y", "yes", "1", ""]:
                exit()
            else:
                os.makedirs(args.save_weights_path)

    if args.metrics_path is not None:
        if not os.path.isdir(args.metrics_path):
            response = input("Metrics path does not exist. Create it? (Y/n) >> ")
            if response.lower() not in ["y", "yes", "1", ""]:
                exit()
            else:
                os.makedirs(args.metrics_path)


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="data_path", metavar="DATA_PATH", help="Path to read examples from.")
    parser.add_argument("-sW", "--save_weights_path", metavar="SAVE_WEIGHTS_PATH", default=None, help="Path to save trained weights to. If no path is specified checkpoints are not saved.")
    parser.add_argument("-sM", "--save_model_path", metavar="SAVE_MODEL_PATH", default=None, help="Path to save trained model to.")
    parser.add_argument("-l", "--load_path", metavar="LOAD_PATH", default=None, help="Path to load trained model from. If no path is specified model is trained from scratch.")
    parser.add_argument("-m", "--metrics-path", metavar="METRICS_PATH", default=None, help="Path to save additional performance metrics to (for debugging purposes).")
    parser.add_argument("--PCA", metavar="PCA", default=False, help="If true, a PCA plot is saved.")
    parser.add_argument("--TSNE", metavar="TSNE", default=False, help="If true, a TSNE plot is saved.")

    args = parser.parse_args()
    parse_args(args)

    X_shape, y_shape = utils.get_shapes(args.data_path, "train")

    # Build model
    input_shape = X_shape[1:]
    tower_model = build_tower_cnn_model(input_shape)  # single input model
    triplet_model = build_triplet_model(input_shape, tower_model)  # siamese model
    if args.load_path is not None:
        triplet_model.load_weights(args.load_path)

    # Setup callbacks for early stopping and model saving
    callback_list = setup_callbacks(args.save_weights_path)

    # Compile model
    adam = Adam(lr=LEARNING_RATE)
    triplet_model.compile(optimizer=adam, loss='mean_squared_error')
    tower_model.predict(np.zeros((1,) + input_shape))  # predict on some random data to activate predict()

    # Initializate online triplet generators
    training_batch_generator = OnlineTripletGenerator(args.data_path, "train", tower_model, batch_size=BATCH_SIZE, triplet_mode="batch_all")
    validation_batch_generator = utils.DataGenerator(args.data_path, "valid", batch_size=BATCH_SIZE)

    triplet_model.fit_generator(generator=training_batch_generator, validation_data=validation_batch_generator,
                                callbacks=callback_list, epochs=EPOCHS)

    # Save weights
    if args.save_weights_path is not None:
        triplet_model.save_weights(args.save_weights_path + "final_weights.hdf5")

    # Save model
    if args.save_model_path is not None:
        tower_model.save(args.save_model_path + "tower_model.hdf5")
        triplet_model.save(args.save_model_path + "triplet_model.hdf5")

    # Plot PCA/TSNE
    # TODO: add function in util that reads a specified number of random samples from a dataset.
    if args.PCA is not False or args.TSNE is not False:
        X_valid, y_valid = utils.load_examples(args.data_path, "train")
        X, Y = utils.shuffle_data(X_valid[:, :, :], y_valid[:, :], one_hot_labels=True)
        X = X[:5000, :, :]
        Y = Y[:5000, :]
        X = tower_model.predict(X)
        if args.PCA:
            utils.plot_with_PCA(X, Y)
        if args.TSNE:
            utils.plot_with_TSNE(X, Y)


if __name__ == "__main__":
    main()