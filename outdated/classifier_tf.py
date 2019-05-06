import tensorflow as tf

INPUT_LENGTH = 30  # length of feature vector
OUTPUT_LENGTH = 150  # number of different users to classify

NUM_HIDDEN = 128  # number of hidden units in LSTM


def cnn_rnn_net(X, n_classes, dropout_rate, training):
    """
    Builds classifier model (CNN + RNN)
    """

    conv = tf.layers.conv2d(inputs=X, filters=32,
                            kernel_size=[2, 0], activation=tf.nn.relu)

    embeddings = tf.contrib.layers.flatten(conv)
    embeddings = tf.expand_dims(embeddings, -1)

    # lstm_cell = tf.keras.layers.LSTMCell(units=NUM_HIDDEN)
    # rnn = tf.keras.layers.RNN(lstm_cell, embedding, dtype=tf.float32)

    # outputs = rnn(embedding)

    logits = tf.layers.dense(embeddings, units=OUTPUT_LENGTH)

    return logits


def load_data():
    """
    Loads training data.
    """
    X = None
    y = None

    return X, y


def next_batch(X, y):
    """
    Returns the next batch of training data.
    """
    X_batch = X
    y_batch = y

    return X_batch, y_batch


def main():

    # Hyperparameters
    learning_rate = 0.001
    training_epochs = 500

    # Other options
    display_step = 10  # how often to display progress

    # Input placeholders
    X = tf.placeholder(tf.float32, [None, INPUT_LENGTH, 1, 1])
    y = tf.placeholder(tf.float32, [None, OUTPUT_LENGTH, 1, 1])

    # Define network and get output
    logits = cnn_rnn_net(X, n_classes=OUTPUT_LENGTH, dropout_rate=0, training=True)

    # Compute accuracy
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Define loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    loss_mean = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Load the training data
    X_train, y_train = load_data()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, training_epochs+1):

            X_batch, y_batch = next_batch(X_train, y_train)

            sess.run(optimizer.minimize(loss_mean), feed_dict={X: X_batch, y: y_batch})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([optimizer.minimize(loss_mean), accuracy],
                                     feed_dict={X: X_batch, y: y_batch})

                print("Step {}, Minibatch Loss= {:.4f}, Training Accuracy={:.3f}".format(step, loss, acc))

    print("Optimization Finished!")


if __name__ == "__main__":
    main()
