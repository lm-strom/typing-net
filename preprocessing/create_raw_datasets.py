import os
import argparse


def parse_args(args):

    assert os.path.isdir(args.data_path), "Data path does not exist."

    if not os.path.isdir(args.output_path):
        response = input("Output path does not exist. Create it? (Y/n) >> ")
        if response.lower() not in ["y", "yes", "1", ""]:
            exit()
        else:
            os.makedirs(args.output_path)


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="data_path", metavar="DATA_PATH", help="Path to read data from.")
    parser.add_argument(dest="output_path", metavar="MODEL_PATH", help="Path to create new datasets in.")
    args = parser.parse_args()
    parse_args(args)

    # Set of users to try
    sets = [list(range(30, 60)), list(range(60, 90)), list(range(90, 120)), list(range(118, 148))]

    for user_ids in sets:

        assert len(user_ids) == 30, "Set of IDs does not have length 30."

        dest = args.output_path + "30_users_ids_{}_{}/".format(min(user_ids), max(user_ids))
        os.makedirs(dest)
        for user_id in user_ids:
            n_digits = len(str(user_id))
            n_zeros = 3 - n_digits
            os.system("cp {}*.txt {}".format(args.data_path + n_zeros * str(0) + str(user_id), dest))

        print("Successfully created dataset: {}".format(dest))


if __name__ == "__main__":
    main()
