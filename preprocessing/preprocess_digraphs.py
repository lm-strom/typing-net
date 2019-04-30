import os
import sys
import hashlib
from tqdm import tqdm

KEYS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M", "Space", "LShiftKey", "RShiftKey", "Back", "Oemcomma", "OemPeriod", "NumPad0", "NumPad1", "NumPad2", "NumPad3", "NumPad4", "NumPad5", "NumPad6", "NumPad7", "NumPad8", "NumPad9", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]


def parse_raw_data(read_path, write_path, session_fraction=1, special_keys=True):

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
			print file_name

			openDigraphs = {} # first_key||second_key : [first_PT, first_RT, second_PT, second_RT]
			lastPressedKey = None # (key, PT, RT)

			for i, line in enumerate(file):

				if i >= n_lines*session_fraction:
					break

				key, action, time = line.split()

				if not special_keys and key not in KEYS:
					continue

				if action == "KeyDown":
					if lastPressedKey != None:
						digraph = lastPressedKey[0] + "+" + key
						openDigraphs[digraph] = [lastPressedKey[1], lastPressedKey[2], time, None]
					lastPressedKey = [key, time, None]

				elif action == "KeyUp":

					if bool(openDigraphs) == False and lastPressedKey != None:
						lastPressedKey[2] = time

					deleteSet = []
					for key_iterator in openDigraphs:

						if key not in key_iterator.split("+"):
							continue

						#update digraph containing key
						firstOrSecond = int(key_iterator.split("+")[1] == key)
						openDigraphs[key_iterator][1+2*firstOrSecond] = time

						#if digraph finished
						if None not in openDigraphs[key_iterator]:

							# Append to output
							ht1 = int(openDigraphs[key_iterator][1]) - int(openDigraphs[key_iterator][0])
							ht2 = int(openDigraphs[key_iterator][3]) - int(openDigraphs[key_iterator][2])
							ptp = int(openDigraphs[key_iterator][2]) - int(openDigraphs[key_iterator][0])
							rtp = int(openDigraphs[key_iterator][2]) - int(openDigraphs[key_iterator][1])
							key1 = key_iterator.split("+")[0]
							key2 = key_iterator.split("+")[1]
							output.append((key1, key2, ht1, ht2, ptp, rtp))

							#delete digraph from openDigraphs
							deleteSet.append(key_iterator)
					for digraph in deleteSet:
						del openDigraphs[digraph]

			# Write processed data to file
			try:
				os.makedirs(write_file[:-8])
			except:
				pass
			with open(write_file, "w+") as file:
				for entry in output:
					file.write(entry[0] + " " + entry[1] + " " + str(entry[2]) + " " + str(entry[3]) + " " + str(entry[4]) + " " + str(entry[5]) + "\n")


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