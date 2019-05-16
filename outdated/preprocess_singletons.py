import os
import sys
import hashlib

KEYS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M", "Space", "LShiftKey", "RShiftKey", "Back", "Oemcomma", "OemPeriod", "NumPad0", "NumPad1", "NumPad2", "NumPad3", "NumPad4", "NumPad5", "NumPad6", "NumPad7", "NumPad8", "NumPad9", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]


def parse_raw_data(read_path, write_path, session_fraction=1, special_keys=False):

	filenames = list()
	for (dirpath, dirnames, _filenames) in os.walk(read_path):
		filenames += [os.path.join(dirpath, file) for file in _filenames]

	for file_name in filenames:

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

			pressedDown = {} # (key : time)
			lastPressTime = None

			for i, line in enumerate(file):

				if i >= n_lines*session_fraction:
					break

				key, action, time = line.split()

				if not special_keys and key not in KEYS:
					continue

				if action == "KeyDown":
					pressedDown[key] = time

					if lastPressTime != None:
						gapTime = int(time) - int(lastPressTime)
						if gapTime < 1000:
							output.append(("0", gapTime))

					lastPressTime = time

				elif action == "KeyUp":
					try:
						keyHash = str(int(hashlib.md5(str.encode(key)).hexdigest()[0:5], 16))
						output.append((keyHash, int(time) - int(pressedDown.pop(key))))
					except:
						pass

			# Write processed data to file
			try:
				os.makedirs(write_file[:-8])
			except:
				pass
			with open(write_file, "w+") as file:
				for entry in output:
					file.write(entry[0] + " " + str(entry[1]) + "\n")


def main():

	if len(sys.argv) < 3:
		print("Missing required arguments: input_path output_path [sesh_fraction=0-1] [special_keys=1/0]")
		exit()

	inputDataPath = sys.argv[1]
	processedDataPath = sys.argv[2]

	# Check if path for preprocessed data exists
	if os.path.exists(processedDataPath):
		ans = raw_input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
		if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
			exit()

	# Creates paths for the preprocessed data
	if not os.path.exists(processedDataPath):
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