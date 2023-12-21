#!/bin/bash
import json


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
			if row_num < 2:
				continue
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
	distance = 16

	if cl1["repre_id"] == cl2["repre_id"]:
		distance = 0
	elif cl1["FA"] == cl2["FA"]:
		distance = 1
	elif cl1["SF"] == cl2["SF"]:
		distance = 2
	elif cl1["CF"] == cl2["CF"]:
		distance = 4
	elif cl1["CL"] == cl2["CL"]:
		distance = 8
	elif cl1["TP"] == cl2["TP"]:
		distance = 8

	return distance


def build_dataset(sample_proteins, sample_protein_info, dataset_file):
	thekeys = list(sample_proteins.keys())

	with open(dataset_file, 'w') as fout:
		fout.write(f"protein1_pdb\tprotein2_pdb\tdistance\tprotein1_seq\tprotein2_seq\n")
		for protein1 in thekeys:
			for protein2 in thekeys[1:]:
				cl1 = sample_protein_info[protein1]
				cl2 = sample_protein_info[protein2]
				distance = measure_distance(cl1, cl2)

				fout.write(f"{protein1}\t{protein2}\t{distance}\t{sample_proteins[protein1]}\t{sample_proteins[protein2]}\t{json.dumps(cl1)}\t{json.dumps(cl2)}\n")


def main():
	protein_info_file = "../generated_data/scop-protein-whole-info.txt"
	sample_protein_file = "../generated_data/protein_samples_seq.txt"
	dataset_file = "../generated_data/sample_proteins_dataset.txt"


	sample_proteins, sample_protein_info = \
		parse_class_info(protein_info_file, sample_protein_file)
	# print(f"{json.dumps(sample_proteins)}\n\n{json.dumps(sample_protein_info)}")

	build_dataset(sample_proteins, sample_protein_info, dataset_file)



if __name__ == "__main__":
    main()

