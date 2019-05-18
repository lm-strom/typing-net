"""
Builds a siamese CNN and trains it to embed typing data from same user to be similar,
and typing data from different users to be dissimilar.

Adapted from:
https://github.com/divyashan/time_series/blob/master/models/supervised/siamese_triplet_keras.py
"""

import os
import signal
import argparse

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import keras.backend as K

import utils

# Constants
PERIOD = 10

# Parameters
ALPHA = 1  # Triplet loss threshold
LEARNING_RATE = 1e-6
EPOCHS = 1000
BATCH_SIZE = 50

# Global variables
stop_flag = False  # Flag to indicate that training was terminated early
training_complete = False  # Flag to indicate that training is complete


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
        model_checkpoint = ModelCheckpoint(save_path + "_class_model_{epoch:02d}_{val_loss:.2f}.hdf5", monitor="val_loss", save_best_only=True, verbose=1, period=10) # Save model every 100 epochs
        callback_list.append(model_checkpoint)

    return callback_list


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
    return K.maximum(_euclidean_distance([A, P]) - _euclidean_distance([A, N]) + ALPHA, 0.0)


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
    pca = PCA(n_components=2)

    X_embedded = StandardScaler().fit_transform(X_embedded)
    X_embedded = pca.fit_transform(X_embedded)

    y = np.array(utils.one_hot_to_index(y))

    import matplotlib.pyplot as plt
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y)
    plt.show()


def plot_with_TSNE(X_embedded, y):
    """
    Applies t-SNE (with n_components=2) to X_embedded and plots
    resulting (x_1, x_2) in 2D, with color indicating class.

    Scikit-learn has t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    tsne = TSNE(n_components=2, verbose=1)

    X_embedded = StandardScaler().fit_transform(X_embedded)
    X_embedded = tsne.fit_transform(X_embedded)

    y = np.array(utils.one_hot_to_index(y))

    import matplotlib.pyplot as plt
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y)
    plt.show()


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
    args = parser.parse_args()
    parse_args(args)


    # Load training triplets and validation triplets
    X_train_anchors, y_train_anchors = utils.load_examples(args.data_path, "train_anchors")
    X_train_positives, _ = utils.load_examples(args.data_path, "train_positives")
    X_train_negatives, _ = utils.load_examples(args.data_path, "train_negatives")
    X_valid_anchors, y_valid_anchors = utils.load_examples(args.data_path, "valid_anchors")
    X_valid_positives, _ = utils.load_examples(args.data_path, "valid_positives")
    X_valid_negatives, _ = utils.load_examples(args.data_path, "valid_negatives")

    # Build model
    input_shape = X_train_anchors.shape[1:]
    tower_model = build_tower_cnn_model(input_shape)  # single input model
    triplet_model = build_triplet_model(input_shape, tower_model)  # siamese model
    if args.load_path is not None:
    	triplet_model.load_weights(args.load_path)

    # Setup callbacks for early stopping and model saving
    callback_list = setup_callbacks(args.save_weights_path)

    # Compile model
    adam = Adam(lr=LEARNING_RATE)
    triplet_model.compile(optimizer=adam, loss='mean_squared_error')

    # Create dummy y = 0 (since output of siamese model is triplet loss)
    y_train_dummy = np.zeros((X_train_anchors.shape[0],))
    y_valid_dummy = np.zeros((X_valid_anchors.shape[0],))

    # Train the model
    triplet_model.fit([X_train_anchors, X_train_positives, X_train_negatives], y_train_dummy, validation_data=([X_valid_anchors, X_valid_positives, X_valid_negatives], y_valid_dummy), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callback_list)
    global training_complete
    training_complete = True

    # Save weights
    if args.save_weights_path is not None:
        triplet_model.save_weights(args.save_weights_path + "final_weights.hdf5")

    # Save model
    if args.save_model_path is not None:
        tower_model.save(args.save_model_path + "model.hdf5")

    # Plot PCA/TSNE
    X, Y = utils.shuffle_data(X_valid_anchors, y_valid_anchors, one_hot_labels=True)
    X = X[:5000,:,:]
    Y = Y[:5000,:]
    X = tower_model.predict(X)
    plot_with_TSNE(X, Y)



    # Other things that may be useful later:

    # Evaluation function found in util.py of the Jiffy git repo
    # print(evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test))

    # Training using fit_generator (probably not necessary for us because data is limited)
    # triplet_model.fit_generator(gen_batch(X_train, tr_trip_idxs, batch_size, dummy_y), epochs=1, steps_per_epoch=n_batches_per_epoch)


if __name__ == "__main__":
    main()
