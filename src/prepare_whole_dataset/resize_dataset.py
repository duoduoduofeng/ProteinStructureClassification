import sys
import json
import random
import os


def count_by_dis(input_file):
	stat_dict = {}

	with open(input_file, 'r') as fin:
		row_num = 0
		for line in fin:
			row_num += 1

			if row_num == 1:
				continue

			parts = line.strip().split("\t")
			dis = int(parts[2])
			if dis not in stat_dict:
				stat_dict[dis] = 0
			stat_dict[dis] += 1
	
	sorted_dict = dict(sorted(stat_dict.items(), key=lambda item: item[1], reverse = True))
	return sorted_dict


# The amount of pairs with distance 8 is too large, sample it to the second large set size.
def sample_largest_set(input_file, theseed = 2024):
	random.seed(theseed)

	output_file = f"{input_file}.tailored"
	
	stat_dict = count_by_dis(input_file)
	print(f"The statistic by distance for the input file: {json.dumps(stat_dict)}.")

	# The key with the largest amount
	largest_key = list(stat_dict.keys())[0]
	second_largest_key = list(stat_dict.keys())[1]

	sample_size = int(stat_dict[second_largest_key] / 2)
	probability = float(sample_size / stat_dict[largest_key])

	with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
		row_num = 0
		for line in fin:
			row_num += 1

			if row_num == 1:
				fout.write(line)
				continue

			parts = line.strip().split("\t")
			dis = int(parts[2])

			if dis != largest_key:
				fout.write(line)
			else: # sample from largest_key to second_largest_key
				if sample_size == 0:
					continue
				if random.random() < probability:
					fout.write(line)
					sample_size -= 1

	os.rename(input_file, f"{input_file}.original")
	os.rename(output_file, input_file)

	new_stat_dict = count_by_dis(input_file)
	print(f"Tailored the original file. The statistic is now: {json.dumps(new_stat_dict)}")


if __name__ == "__main__":
	input_file = sys.argv[1]
	theseed = 2024
	sample_largest_set(input_file, theseed)

