import os
import argparse


def parse_args(args):
    """
    Checks that input args are valid.
    """
    assert os.path.isfile(args.data_path), "The specified data file does not exist."
    assert os.path.isfile(args.model_path), "The specified model file does not exist."


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="data_path", metavar="data_PATH", help="Path to read data from.")
    parser.add_argument(dest="model_path", metavar="MODEL_PATH", help="Path to read model from.")
    args = parser.parse_args()
    parse_args(args)


    # Load model
    with CustomObjectScope({'_euclidean_distance': cnn_siamese_online._euclidean_distance,
                            'ALPHA': cnn_siamese_online.ALPHA}):
        tower_model = load_model(args.model_path)
        tower_model.compile(optimizer='adam', loss='mean_squared_error')  # Model was previously not compiled

if __name__ == "__main__":
    main()
