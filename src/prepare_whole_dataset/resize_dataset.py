import sys
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np


def bin_stat(my_list):
	bins = [0, 100, 200, 300, 400, 500, 1000, 2000, 5000, 100000]
	hist, bin_edges = np.histogram(my_list, bins=bins)

	for i in range(len(hist)):
		print(f"Bucket {i+1}: Frequency: {hist[i]}, Boundaries: [{bin_edges[i]}, {bin_edges[i+1]}]")


def count_by_dis(input_file):
	stat_dict = {}
	seq1_len = []
	seq2_len = []

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

			seq1_len.append(len(parts[3]))
			seq2_len.append(len(parts[4]))
	
	sorted_dict = dict(sorted(stat_dict.items(), key=lambda item: item[1], reverse = True))

	print(f"Start bin stat for seq1.")
	bin_stat(seq1_len)
	print(f"Start bin stat for seq2.")
	bin_stat(seq2_len)

	return sorted_dict


# The amount of pairs with distance 8 is too large, sample it to the second large set size.
def sample_largest_set(input_file, output_file, theseed = 2024):
	random.seed(theseed)
	
	stat_dict = count_by_dis(input_file)
	print(f"The statistic by distance for the input file: {json.dumps(stat_dict)}.\n")

	# The key with the largest amount
	largest_key = list(stat_dict.keys())[0]
	second_largest_key = list(stat_dict.keys())[1]

	sample_size = stat_dict[second_largest_key]
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

			# Filter too long sequences.
			if len(parts[3]) > 500 or len(parts[4]) > 500:
				continue

			if dis != largest_key:
				fout.write(line)
			else: # sample from largest_key to second_largest_key
				if sample_size == 0:
					continue
				if random.random() < probability:
					fout.write(line)
					sample_size -= 1

	new_stat_dict = count_by_dis(output_file)
	print(f"Tailored the original file {input_file} to {output_file}. \nThe statistic is now: {json.dumps(new_stat_dict)}")


if __name__ == "__main__":
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	theseed = 2024
	sample_largest_set(input_file, output_file, theseed)

