import os
import shutil
import hashlib
import argparse
import numpy as np

from tqdm import tqdm

KEYS = [["", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D0", "", "", "", "Back"], 
    ["", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"], 
    ["", "A", "S", "D", "F", "G", "H", "J", "K", "L",],
    ["LShiftKey", "Z", "X", "C", "V", "B", "N", "M", "Oemcomma", "OemPeriod", "", "RShiftKey"], 
    ["","","","","","Space", "Space", "Space", "Space", "Space" ]]

KEYS_FLAT = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A",
        "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C",
        "V", "B", "N", "M", "Space", "LShiftKey", "RShiftKey",
        "Back", "Oemcomma", "OemPeriod", "D0", "D1", "D2", "D3",
        "D4", "D5", "D6", "D7", "D8", "D9"]

def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))

def parse_raw_data(read_path, write_path):

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

                if key not in KEYS_FLAT:
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
                        keyDistance = np.sum(np.absolute(np.array(index_2d(KEYS, key1)) - np.array(index_2d(KEYS, key2))))
                        output.append((keyDistance, ht1, ht2, ptp, rtp))
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
                    file.write(str(entry[0]) + " " + str(entry[1]) + " " + str(entry[2]) + " " + str(entry[3]) + " " + str(entry[4]) + "\n")
                file.close()
        except:
            with open(write_file, "w+") as file:
                for entry in output:
                    file.write(str(entry[0]) + " " + str(entry[1]) + " " + str(entry[2]) + " " + str(entry[3]) + " " + str(entry[4]) + "\n")
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
    parse_raw_data(args.input_path, args.output_path)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()
