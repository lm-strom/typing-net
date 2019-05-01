import os
import shutil
import sys
import hashlib
from tqdm import tqdm

KEYS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M", "Space", "LShiftKey", "RShiftKey", "Back", "Oemcomma", "OemPeriod", "NumPad0", "NumPad1", "NumPad2", "NumPad3", "NumPad4", "NumPad5", "NumPad6", "NumPad7", "NumPad8", "NumPad9", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]

NUM_HASHES = 1000

def parse_raw_data(read_path, write_path, session_fraction=1, special_keys=False, hash_keys=True):

	filenames = list()
	for (dirpath, dirnames, _filenames) in os.walk(read_path):
		filenames += [os.path.join(dirpath, file) for file in _filenames]

	for file_name in tqdm(filenames):

		filepath = file_name.split("/")
		if filepath[-1][0] == ".":
			continue
		subPath = ""
		for entry in filepath[-3:-1]:
			subPath += "/" + entry
		user_id = filepath[-1][:3]
		write_file = write_path + subPath + "/" + user_id + ".txt"

		n_lines = len(open(file_name).readlines())
		output = []

		with open(file_name, "r") as file:

			pressedKeys = [] # [[key, PT, RT], ...]

			for i, line in enumerate(file):

				if i >= n_lines*session_fraction:
					break

				key, action, time = line.split()

				if not special_keys and key not in KEYS:
					continue

				if action == "KeyDown":
					pressedKeys.append([key, time, None])

				elif action == "KeyUp":
					for pressedKey in pressedKeys[::-1]:
						if pressedKey[0] == key:
							if pressedKey[2] == None:
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
							key1Hash = str(int(hashlib.md5(str.encode(key1)).hexdigest()[0:5] % NUM_HASHES, 16))
							key2Hash = str(int(hashlib.md5(str.encode(key2)).hexdigest()[0:5] % NUM_HASHES, 16))
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



def main():

	if len(sys.argv) < 3:
		print("Missing required arguments: input_path output_path [sesh_fraction=0-1] [special_keys=1/0]")
		exit()

	inputDataPath = sys.argv[1]
	processedDataPath = sys.argv[2]

	# Check if path for preprocessed data exists
	if os.path.exists(processedDataPath):
		ans = input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
		if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
			exit()

	# Creates fresh path for the preprocessed data
	if os.path.exists(processedDataPath):
		if "processed_data" not in processedDataPath:
			print("Processed data path must include \"processed_data\" as a precaution.")
		else:
			shutil.rmtree(processedDataPath)
	os.mkdir(processedDataPath)


	if len(sys.argv) == 3:
		parse_raw_data(inputDataPath, processedDataPath)
	elif len(sys.argv) == 4:
		try:
			seshFrac = float(sys.argv[3])
		except:
			print("sesh_fraction needs to be a float between 0 and 1")
			exit()
		parse_raw_data(inputDataPath, processedDataPath, seshFrac)
	else:
		try:
			seshFrac = float(sys.argv[3])
		except:
			print("sesh_fraction needs to be a float between 0 and 1")
			exit()
		try:
			specKeys = int(sys.argv[4])
		except:
			print("special_keys needs to be a integer, either 0 or 1")
			exit()
		parse_raw_data(inputDataPath, processedDataPath, seshFrac, specKeys)

	print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()
