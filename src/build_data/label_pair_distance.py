#!/bin/bash
import json
from itertools import combinations
import random
import math


# narrow the protein info parsing objective into the sample proteins
def parse_class_info(protein_info_file, sample_protein_file):
	protein_info_field_count = 14
	sample_info_field_count = 3

	sample_proteins = {}
	sample_protein_info = {}

	# Load the sample proteins
	row_num = 0
	with open(sample_protein_file, 'r') as pifin:
		for line in pifin:
			row_num += 1
			# if row_num < 2:
			#	continue
			parts = line.strip().split('\t')
			if len(parts) == sample_info_field_count:
				pdb_id, chain_id, seq = parts
				sample_proteins[f"{pdb_id}_{chain_id}"] = seq
	print(f"Loaded the sample proteins.")

	# Parse the protein information
	row_num = 0
	with open(protein_info_file, 'r') as spfin:
		for line in spfin:
			row_num += 1
			if row_num < 2:
				continue
			parts = line.strip().split(' ')
			if len(parts) == protein_info_field_count:
				repre_id = parts[0]
				pdb_id = parts[1]
				chain_id = parts[2]
				thekey = f"{pdb_id}_{chain_id}"

				# parse only the sample proteins
				if thekey not in sample_proteins:
					continue

				class_info_dict = {}
				class_info_dict["repre_id"] = repre_id

				class_info = parts[-1].split(",")
				for cl in class_info:
					clparts = cl.split("=")
					class_info_dict[clparts[0]] = clparts[1]
				sample_protein_info[thekey] = class_info_dict
	print(f"Loaded the classification information of the sample proteins.")

	return sample_proteins, sample_protein_info


def measure_distance(cl1, cl2):
	distance = 1

	if cl1["repre_id"] == cl2["repre_id"]:
		distance = 0
	elif cl1["FA"] == cl2["FA"]:
		distance = 1/16
	elif cl1["SF"] == cl2["SF"]:
		distance = 2/16
	elif cl1["CF"] == cl2["CF"]:
		distance = 4/16
	elif cl1["CL"] == cl2["CL"]:
		distance = 8/16
	elif cl1["TP"] == cl2["TP"]:
		distance = 8/16

	return distance


def build_dataset(sample_proteins, sample_protein_info, dataset_file, cut_percent, theseed = None):
	thekeys = list(sample_proteins.keys())

	# get all 2-item combination
	all_pairs = list(combinations(thekeys, 2))
	pair_count = len(all_pairs)
	print(f"There are {pair_count} pairs of protein selected in total.")

	dis_dict = {}
	dis_8_dict = {}
	# Calculate the distance of each protein pairs.
	for pro1, pro2 in all_pairs:
		cl1 = sample_protein_info[pro1]
		cl2 = sample_protein_info[pro2]
		distance = measure_distance(cl1, cl2)

		pair_key = f"{pro1}||{pro2}"

		if distance == 8/16:
			dis_8_dict[pair_key] = distance
		else:
			dis_dict[pair_key] = distance

	# Sample the pairs in the same class/protein type (distance = 8)
	random.seed(theseed)
	dis_8_count = len(dis_8_dict)
	print(f"Before sampling, there are {dis_8_count} protein pairs with distance 8 in total.")
	
	cut_size = math.floor(dis_8_count * cut_percent)
	sampled_items = random.sample(list(dis_8_dict.items()), cut_size)
	# Convert the sampled items back to a dictionary
	sampled_dis_8_dict = dict(sampled_items)
	print(f"After sampling, there are {len(sampled_dis_8_dict)} protein pairs with distance 8 in total.")

	# save dataset
	excluded_pair_count = 0
	with open(dataset_file, 'w') as fout:
		fout.write(f"protein1_pdb\tprotein2_pdb\tdistance\tprotein1_seq\tprotein2_seq\tprotein1_classification\tprotein2_classfication\n")
		for pro1, pro2 in all_pairs:
			thekey = f"{pro1}||{pro2}"
			distance = -1
			if thekey in dis_dict:
				distance = dis_dict[thekey]
			elif thekey in sampled_dis_8_dict:
				distance = sampled_dis_8_dict[thekey]
			
			if distance != -1:
				cl1 = sample_protein_info[pro1]
				cl2 = sample_protein_info[pro2]
				fout.write(f"{pro1}\t{pro2}\t{distance}\t{sample_proteins[pro1]}\t{sample_proteins[pro2]}\t{json.dumps(cl1)}\t{json.dumps(cl2)}\n")
			else:
				excluded_pair_count += 1
		print(f"******* There are {excluded_pair_count} protein pairs excluded in total.")


def main():
	protein_info_file = "../../generated_data/scop-protein-whole-info.txt"
	sample_protein_file = "../../generated_data/protein_samples_seq.txt"
	dataset_file = "../../generated_data/sample_proteins_dataset.txt"
	cut_percent = 0.04
	theseed = 2023


	sample_proteins, sample_protein_info = \
		parse_class_info(protein_info_file, sample_protein_file)
	# print(f"{json.dumps(sample_proteins)}\n\n{json.dumps(sample_protein_info)}")

	build_dataset(sample_proteins, sample_protein_info, dataset_file, cut_percent, theseed)



if __name__ == "__main__":
    main()

