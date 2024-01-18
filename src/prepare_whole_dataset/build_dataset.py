import sys
import json
import random
import copy
import math
from itertools import combinations


"""
Count the frequency of each {pdb_id, chain_id} pair.
"""
def uniq_proteins_count(proteins_file):
	# Dict runs much faster than list
	# {pdb_id: {chain_id, count}}
	uniq_pros = {}

	with open(proteins_file["filename"], 'r') as f2:
		row_num = 0
		for line in f2:
			row_num += 1
			if row_num < proteins_file["start_row_num"]:
				continue

			parts = line.strip().split(' ')
			if len(parts) == proteins_file["fields_count"]:
				pdb_id = parts[1]
				chain_id = parts[2]
				if pdb_id not in uniq_pros:
					new_chains = {}
					new_chains[chain_id] = 1
					uniq_pros[pdb_id] = new_chains
				else:
					cur_chain = uniq_pros[pdb_id]
					if chain_id not in cur_chain:
						cur_chain[chain_id] = 1
					else:
						cur_chain[chain_id] = cur_chain[chain_id] + 1

	print(f"There are {len(uniq_pros)} uniq pdb_id in total.")

	return uniq_pros


"""
Reserve the {pdb_id, chain_id} pairs which occurs exactly twice.
PS, only family representative domains are keeped. 
DOMID REPRESENTED-PDBID REPRESENTED-PDBCHAIN FA-DOMID FA-PDBID FA-PDBREG FA-UNIID FA-UNIREG SF-DOMID SF-PDBID SF-PDBREG SF-UNIID SF-UNIREG SCOPCLA
8000061 2DT5 B 8000061 2DT5 A:2-77 Q5SHS3 2-77 8001519 2DT5 A:4-63 Q5SHS3 4-63 TP=1,CL=1000003,CF=2000145,SF=3000034,FA=4000057
"""
def filter_multi_domains_protein(uniq_pros, uniq_pros_previous_file, merged_info_file):
	# generate an assist file, to help locate the sequence filename.
	row_flags = {}
	with open(uniq_pros_previous_file, 'r') as fin:
		row_num = 0
		for line in fin:
			pdb_id = line.split("\t")[0]
			row_num += 1
			row_flags[pdb_id] = row_num

	reserved_proteins = {}
	with open(merged_info_file, 'r') as fin:
		row_num = 0
		sf_row_count = 0
		for line in fin:
			row_num += 1
			if row_num < 2:
				continue

			parts = line.strip().split(" ")
			if len(parts) == 14:
				rep_id = parts[0]
				pdb_id = parts[1]
				chain_id = parts[2]
				lineage_info = parts[-1]
				# Here filters the proteins which contains not 2 domains.
				if uniq_pros[pdb_id][chain_id] == 2:
					thekey = f"{pdb_id}_{chain_id}"
					info = {}
					info["rep_id"] = rep_id
					info["sequence_file_row_num"] = row_flags[pdb_id]
					# TP=1,CL=1000002,CF=2000148,SF=3000038,FA=4000119
					for thestr in lineage_info.split(","):
						nodename, nodevalue = thestr.split("=")
						info[nodename] = nodevalue
					reserved_proteins[thekey] = info

			elif len(parts) == 3:
				# print(f"**** Line {row_num} is not for family domain but super family's.")
				sf_row_count += 1
				continue
			else:
				print(f"Invalid input, line {row_num} in {merged_info_file}.")
				continue

	print(f"There are {len(reserved_proteins)} proteins reserved. \nAnd there are {sf_row_count} super family lines.\nPrint some samples to check the reserved proteins.")
	# an_count = 0
	# for thekey in reserved_proteins:
	# 	an_count += 1
	# 	if an_count%10000 == 1:
	# 		print(f"{thekey}\t{json.dumps(reserved_proteins[thekey])}")

	return reserved_proteins


"""
Stat the lineage distribution of the reserved proteins.
"""
def stat_lineage(reserved_proteins):
	stats = {
		"TP": {}, 
		"CL": {},
		"CF": {},
		"SF": {}, 
		"FA": {}, 
		"rep_id": {}
	}
	for thekey in reserved_proteins:
		info = reserved_proteins[thekey]
		if info["TP"] != "4":
			continue
		for node in ["rep_id", "TP", "CL", "CF", "SF", "FA"]:
			if info[node] not in stats[node]:
				stats[node][info[node]] = 1
			else:
				stats[node][info[node]] = stats[node][info[node]] + 1
	print(json.dumps(stats))
	return stats


"""
Sample the ranged proteins.
"""
def sample_prots(reserved_proteins, repre_num_threshold = 5, nodename = "TP", node_threshold = "4", theseed = 2024):
	repre_prot_dict = {}
	prots_count = 0
	random.seed(theseed)

	for thekey in reserved_proteins:
		info = reserved_proteins[thekey]
		
		if info[nodename] != node_threshold:
			continue
		
		prots_count += 1
		# Merge the pdb_chain_ids under the same representatives.
		if info["rep_id"] not in repre_prot_dict:
			repre_prot_dict[info["rep_id"]] = [thekey]
		else:
			repre_prot_dict[info["rep_id"]].append(thekey)

	print(f"Before sampling, there are {prots_count} proteins in total.")
	
	excluded_validate_keys = [] # without proteins inside train set.
	validate_keys = [] # with some training proteins inside.
	train_keys = []
	for rep_id in repre_prot_dict:
		if len(repre_prot_dict[rep_id]) > repre_num_threshold:
			cur_prot_list = random.sample(repre_prot_dict[rep_id], repre_num_threshold)
			cur_excluded_prot_list = [item for item in repre_prot_dict[rep_id] if item not in cur_prot_list]
			
			# If there are too many excluded proteins, sample again for the validate set.
			if len(cur_excluded_prot_list) > repre_num_threshold:
				sampled_cur_excluded_prot_list = random.sample(cur_excluded_prot_list, repre_num_threshold)
				excluded_validate_keys.extend(sampled_cur_excluded_prot_list)
			else:
				excluded_validate_keys.extend(cur_excluded_prot_list)
		else:
			cur_prot_list = repre_prot_dict[rep_id]
		
		train_keys.extend(cur_prot_list)

	combined_keys = copy.deepcopy(excluded_validate_keys)
	combined_keys.extend(train_keys)
	validate_keys = random.sample(combined_keys, int(len(combined_keys)/2))

	print(f"After sampling, there are {len(train_keys)} proteins in total.")
	print(f"There are {len(excluded_validate_keys)} excluded proteins in total.")
	print(f"And there are {len(validate_keys)} mixed proteins.")

	return train_keys, validate_keys, excluded_validate_keys


"""
Load the sequences.
"""
def load_seqs(sequences_file_dir):
	# protein_sequences.115001.116000.txt
	seqs = {}
	start_num = 1
	end_num = 1000

	for i in range(0, 116):
		start_row = i * 1000 + start_num
		end_row = i * 1000 + end_num
		dict_name = f"protein_sequences.{start_row}.{end_row}"
		seq_file_name = f"{sequences_file_dir}/{dict_name}.txt"

		# print(f"Open file: {seq_file_name}")

		seqs[dict_name] = {}
		with open(seq_file_name, 'r') as fin:
			row_num = 0
			for line in fin:
				row_num += 1
				parts = line.strip().split("\t")
				if len(parts) != 3:
					print(f"Invalid input on line {row_num} from the file {seq_file_name}")
					continue
				thekey = f"{parts[0]}_{parts[1]}"
				seqs[dict_name][thekey] = parts[2]

	return seqs


def measure_distance(cl1, cl2):
	distance = 16

	if cl1["rep_id"] == cl2["rep_id"]:
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


"""
Build the pair data.
"""
def build_pair(reserved_proteins, keys, seqs, output_file):
	# Obtain the selected proteins' sequences at first
	thesequences = {}
	empty_sequences_count = 0
	for thekey in keys:
		info = reserved_proteins[thekey]
		
		# Locate the sequence file
		sequence_file_row_num = info["sequence_file_row_num"]
		if sequence_file_row_num % 1000 != 0:
			start_row = math.floor(sequence_file_row_num/1000) * 1000 + 1
			end_row = math.floor(sequence_file_row_num/1000) * 1000 + 1000
		else:
			start_row = (int(sequence_file_row_num/1000)-1) * 1000 + 1
			end_row = int(sequence_file_row_num/1000) * 1000

		dict_name = f"protein_sequences.{start_row}.{end_row}"
		if thekey not in seqs[dict_name]:
			empty_sequences_count += 1
			continue
		thesequences[thekey] = seqs[dict_name][thekey]

	print(f"There are {empty_sequences_count} proteins failed to request the PDB sequences.")

	### prepare the fields
	# get all 2-item combination based on the keys with sequences obtained.
	all_pairs = list(combinations(thesequences.keys(), 2))
	pair_count = len(all_pairs)
	print(f"There are {pair_count} pairs of protein selected in total.")

	# Calculate the distance of each protein pairs.
	dis_dict = {}
	for pro1, pro2 in all_pairs:
		cl1 = reserved_proteins[pro1]
		cl2 = reserved_proteins[pro2]
		distance = measure_distance(cl1, cl2)
		
		pair_key = f"{pro1}||{pro2}"
		dis_dict[pair_key] = distance

	with open(output_file, 'w') as fout:
		invalid_1_count = 0
		invalid_2_count = 0
		fout.write(f"protein1_pdb\tprotein2_pdb\tdistance\tprotein1_seq\tprotein2_seq\tprotein1_classification\tprotein2_classfication\n")
		
		for pro1, pro2 in all_pairs:
			pair_key = f"{pro1}||{pro2}"
			if pro1 == pro2:
				invalid_1_count += 1
			elif thesequences[pro1] == thesequences[pro2]:
				invalid_2_count += 1
			else:
				fout.write(f"{pro1}\t{pro2}\t{dis_dict[pair_key]}\t{thesequences[pro1]}\t{thesequences[pro2]}\t{json.dumps(reserved_proteins[pro1])}\t{json.dumps(reserved_proteins[pro2])}\n")
		print(f"invalid_1_count = {invalid_1_count}, invalid_2_count = {invalid_2_count}")


"""
Build the pair data.
"""
def build_pair2(reserved_proteins, keys, seqs, output_file):
	# Obtain the selected proteins' sequences at first
	thesequences = {}
	empty_sequences_count = 0
	for thekey in keys:
		info = reserved_proteins[thekey]
		
		# Locate the sequence file
		sequence_file_row_num = info["sequence_file_row_num"]
		if sequence_file_row_num % 1000 != 0:
			start_row = math.floor(sequence_file_row_num/1000) * 1000 + 1
			end_row = math.floor(sequence_file_row_num/1000) * 1000 + 1000
		else:
			start_row = (int(sequence_file_row_num/1000)-1) * 1000 + 1
			end_row = int(sequence_file_row_num/1000) * 1000

		dict_name = f"protein_sequences.{start_row}.{end_row}"
		if thekey not in seqs[dict_name]:
			empty_sequences_count += 1
			continue
		thesequences[thekey] = seqs[dict_name][thekey]

	print(f"There are {empty_sequences_count} proteins failed to request the PDB sequences.")

	with open(output_file, 'w') as fout:
		invalid_1_count = 0
		invalid_2_count = 0
		fout.write(f"protein1_pdb\tprotein2_pdb\tdistance\tprotein1_seq\tprotein2_seq\tprotein1_classification\tprotein2_classfication\n")
		
		all_keys = list(thesequences.keys())
		pair_count = 0
		for i, pro1 in enumerate(all_keys):
		    for pro2 in all_keys[i+1:]:
		    	pair_count += 1
		    	if pro1 == pro2:
		    		invalid_1_count += 1
		    	elif thesequences[pro1] == thesequences[pro2]:
		    		invalid_2_count += 1
		    	else:
		    		cl1 = reserved_proteins[pro1]
		    		cl2 = reserved_proteins[pro2]
		    		distance = measure_distance(cl1, cl2)
		    		fout.write(f"{pro1}\t{pro2}\t{distance}\t{thesequences[pro1]}\t{thesequences[pro2]}\t{json.dumps(reserved_proteins[pro1])}\t{json.dumps(reserved_proteins[pro2])}\n")
		# 510832666
		print(f"There are {pair_count} pairs of protein selected in total.")
		print(f"invalid_1_count = {invalid_1_count}, invalid_2_count = {invalid_2_count}")


if __name__ == "__main__":
	meta_info = [
	    {
	        "filename": "../../data/scop-represented-structures-latest.txt",
	        "start_row_num": 7,
	        "fields_count": 3,
	        "head": "DOMID REPRESENTED-PDBID REPRESENTED-PDBCHAIN",
	        "example": "8000061 2DT5 B"
	    }
	]

	### Step 1, stat all the proteins
	proteins_file = meta_info[0]
	uniq_pros = uniq_proteins_count(proteins_file)

	### Step 2, 
	uniq_pros_previous_file = "../../generated_data/whole_pdbs/uniq_pdb_list.txt"
	merged_info_file = "../../generated_data/scop-protein-whole-info.txt"
	reserved_proteins = filter_multi_domains_protein(uniq_pros, uniq_pros_previous_file, merged_info_file)

	### Step 3, stat the reserved proteins lineage.
	# stats = stat_lineage(reserved_proteins)

	### Step 4, sample the proteins, mainly sample by representatives.
	train_keys, validate_keys, excluded_validate_keys = \
		sample_prots(reserved_proteins, repre_num_threshold = 5, nodename = "TP", 
			node_threshold = "3", theseed = 2024)

	### Step 5, load the sequences
	sequences_file_dir = "../../generated_data/whole_pdbs/sequences"
	seqs = load_seqs(sequences_file_dir)

	### Step 6, when build pair, remember to filter the pair with the same sequences (or key names)
	# awk -F '\t' '{print $3}' sample_proteins_dataset.train.txt | sort | uniq -c
	train_set_file = "../../generated_data/whole_pdbs/datasets/try_tp_3/sample_proteins_dataset.train.txt"
	build_pair2(reserved_proteins, train_keys, seqs, train_set_file)
	validate_set_file = "../../generated_data/whole_pdbs/datasets/try_tp_3/sample_proteins_dataset.validate.txt"
	build_pair2(reserved_proteins, validate_keys, seqs, validate_set_file)
	excluded_validate_set_file = "../../generated_data/whole_pdbs/datasets/try_tp_3/sample_proteins_dataset.excluded_validate.txt"
	build_pair2(reserved_proteins, excluded_validate_keys, seqs, excluded_validate_set_file)


