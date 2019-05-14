"""
Builds a siamese CNN and trains it to embed typing data from same user to be similar,
and typing data from different users to be dissimilar.

Adapted from:

https://github.com/divyashan/time_series/blob/master/models/supervised/siamese_triplet_keras.py
"""

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import keras.backend as K

# import keras
# from keras.layers import Sequential
# from keras.layers import Activation
# from keras.layers import SimpleRNN
# from keras import initializers

# Parameters
ALPHA = 1  # Triplet loss threshold


def build_tower_cnn_model(input_shape):
    """
    Builds a CNN-model for embedding single examples of data.
    """

    x0 = Input(input_shape, name='Input')

    kernel = 5
    n_channels = [16]
    x = x0
    for i in range(len(n_channels)):
        x = Conv2D(n_channels[i], kernel_size=kernel, strides=(2, 2), activation='relu', padding='same')(x)
        x = MaxPooling2D((5, 1))(x)

    x = Flatten()(x)
    y = Dense(40, name='dense_encoding')(x)

    model = Model(inputs=x0, outputs=y)

    return model


def euclidean_distance(vects):
    """
    Computes euclidean distance between tuple of vectors.
    """
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    """
    Wat?
    """
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def triplet_distance(vects):
    """
    Computes triplet loss for single triplet.
    """
    a, s, d = vects
    return euclidean_distance([a, s]) - euclidean_distance([a, d]) + ALPHA


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

    distance = Lambda(triplet_distance, output_shape=eucl_dist_output_shape)([x_A, x_B, x_C])

    model = Model([input_A, input_B, input_C], distance, name='siamese')

    return model


def main():

    X_train, y_train = np.array([]), np.array([])  # placeholder - training set of single examples
    X_test, y_test = np.array([]), np.array([])  # placeholder - test set of single examples
    input_shape = (1, 2)  # placeholder - shape of single input

    adam = Adam(lr=0.000001)

    tower_model = build_tower_cnn_model(input_shape)  # single input model
    triplet_model = build_triplet_model(input_shape, tower_model)  # siamese model

    triplet_model.compile(optimizer=adam, loss='mean_squared_error')

    # triplet_model.fit([tr_pairs[:,0,:,:], tr_pairs[:,1,:,:]], tr_y, epochs=100, batch_size=50, validation_split=.05)

    # Using fit_generator (probably not necessary for us because data is limited)
    # triplet_model.fit_generator(gen_batch(X_train, tr_trip_idxs, batch_size, dummy_y), epochs=1, steps_per_epoch=n_batches_per_epoch)

    train_embedding = tower_model.predict(X_train)
    test_embedding = tower_model.predict(X_test)

    # Evaluation function found in util.py of the git repo (might use later)
    # print(evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test))


if __name__ == "__main__":
    main()
