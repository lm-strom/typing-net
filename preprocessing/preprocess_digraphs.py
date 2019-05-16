import os
import shutil
import hashlib
import argparse

from tqdm import tqdm

KEYS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A",
        "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C",
        "V", "B", "N", "M", "Space", "LShiftKey", "RShiftKey",
        "Back", "Oemcomma", "OemPeriod", "NumPad0", "NumPad1",
        "NumPad2", "NumPad3", "NumPad4", "NumPad5", "NumPad6",
        "NumPad7", "NumPad8", "NumPad9", "D0", "D1", "D2", "D3",
        "D4", "D5", "D6", "D7", "D8", "D9"]

NUM_HASHES = 1000


def parse_raw_data(read_path, write_path, special_keys=False, hash_keys=True):

    filenames = list()
    for (dirpath, dirnames, _filenames) in os.walk(read_path):
        filenames += [os.path.join(dirpath, file) for file in _filenames]

    for file_name in tqdm(filenames):

        filepath = file_name.split("/")
        if filepath[-1][0] == ".":
            continue
        subPath = ""

        user_id = filepath[-1][:3]
        write_file = write_path + subPath + "/" + user_id + ".txt"

        output = []

        with open(file_name, "r") as file:

            pressedKeys = []  # [[key, PT, RT], ...]

            for line in file:

                key, action, time = line.split()

                if not special_keys and key not in KEYS:
                    continue

                if action == "KeyDown":
                    pressedKeys.append([key, time, None])

                elif action == "KeyUp":
                    for pressedKey in pressedKeys[::-1]:
                        if pressedKey[0] == key:
                            if pressedKey[2] is None:
                                pressedKey[2] = time
                            else:
                                break

            for i in range(len(pressedKeys)):
                if i == len(pressedKeys) - 1:
                    break
                try:
                    ht1 = int(pressedKeys[i][2]) - int(pressedKeys[i][1])
                    ht2 = int(pressedKeys[i+1][2]) - int(pressedKeys[i+1][1])
                    ptp = int(pressedKeys[i+1][1]) - int(pressedKeys[i][1])
                    rtp = int(pressedKeys[i+1][1]) - int(pressedKeys[i][2])
                    key1 = pressedKeys[i][0]
                    key2 = pressedKeys[i+1][0]
                    if ptp < 1000 and abs(rtp) < 1000:
                        if hash_keys:
                            key1Hash = str(int(hashlib.md5(str.encode(key1)).hexdigest()[0:5], 16) % NUM_HASHES)
                            key2Hash = str(int(hashlib.md5(str.encode(key2)).hexdigest()[0:5], 16) % NUM_HASHES)
                            output.append((key1Hash, key2Hash, ht1, ht2, ptp, rtp))
                        else:
                            output.append((key1, key2, ht1, ht2, ptp, rtp))
                except:
                    pass

        # Write processed data to file
        try:
            os.makedirs(write_file[:-8])
        except:
            pass
        try:
            with open(write_file, "a") as file:
                for entry in output:
                    file.write(entry[0] + " " + entry[1] + " " + str(entry[2]) + " " + str(entry[3]) + " " + str(entry[4]) + " " + str(entry[5]) + "\n")
                file.close()
        except:
            with open(write_file, "w+") as file:
                for entry in output:
                    file.write(entry[0] + " " + entry[1] + " " + str(entry[2]) + " " + str(entry[3]) + " " + str(entry[4]) + " " + str(entry[5]) + "\n")
                file.close()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read raw typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write processed data to")
    parser.add_argument("-s", "--special_keys", metavar="SPECIAL_KEYS", type=str2bool, nargs="?",
                        default="False", help="Whether to include special keys (Y/n)")
    args = parser.parse_args()

    # Verify that input path exists
    assert os.path.exists(args.input_path), "Specified input path does not exist."

    # Check if path for preprocessed data exists
    if os.path.exists(args.output_path):
        ans = input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
        if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
            exit()

    # Creates fresh path for the preprocessed data
    if os.path.exists(args.output_path):
        if "processed_data" not in args.output_path:
            print("Processed data path must include \"processed_data\" as a precaution.")
        else:
            shutil.rmtree(args.output_path)
    os.mkdir(args.output_path)

    # Process the data
    parse_raw_data(args.input_path, args.output_path, args.special_keys)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()
