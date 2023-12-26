#!/bin/usr

# {
# 	"family_id": { # 4000088
# 		"prot_count": 100,
# 		"repre_id": { # 8000693
# 			"repre_prot_count": 10,
# 			"prot_ids": [(prot_pdbid, prot_pdbchainid),
# 						 ()]
# 		}
# 	}
# }
import json
import math
import random

def count_family(protein_file):
	row_num = 0
	fields_count = 14
	no_fa_count = 0

	fa_pros = {}

	with open(protein_file, 'r') as fin:
		for line in fin:
			row_num += 1
			if row_num < 2:
				continue

			# head: 
			# DOMID REPRESENTED-PDBID REPRESENTED-PDBCHAIN \
			# FA-DOMID FA-PDBID FA-PDBREG FA-UNIID FA-UNIREG SF-DOMID SF-PDBID SF-PDBREG SF-UNIID SF-UNIREG SCOPCLA
			parts = line.strip().split(' ')
			if len(parts) == fields_count:
				repre_id = parts[0]
				prot_pdbid = parts[1]
				prot_pdbchainid = parts[2]
				family_id = parts[-1].split("=")[-1]

				if family_id not in fa_pros:
					fa_pros[family_id] = {
						"family_id": family_id,
						"prot_count": 1,
						repre_id: {
							"repre_prot_count": 1,
							"prot_ids": [(prot_pdbid, prot_pdbchainid)]
						}
					}
				else:
					ori_fa_info = fa_pros[family_id]
					fa_pros[family_id]["prot_count"] = ori_fa_info["prot_count"] + 1
					if repre_id not in ori_fa_info:
						fa_pros[family_id][repre_id] = {
							"repre_prot_count": 1,
							"prot_ids": [(prot_pdbid, prot_pdbchainid)]
						}
					else:
						ori_repre_info = fa_pros[family_id][repre_id]
						fa_pros[family_id][repre_id]["repre_prot_count"] = \
							ori_repre_info["repre_prot_count"] + 1
						fa_pros[family_id][repre_id]["prot_ids"].append((prot_pdbid, prot_pdbchainid))

			else:
				no_fa_count += 1

	# sort the family dictionary by count
	sorted_fa_pros = dict(sorted(fa_pros.items(), 
							  key=lambda item: item[1]["prot_count"], reverse = True))

	return sorted_fa_pros


def dump_fa_pros(sorted_fa_pros, output_file):
	with open(output_file, 'w') as fout:
		for fa_id in sorted_fa_pros:
			fa_item = sorted_fa_pros[fa_id]
			prot_count = fa_item["prot_count"]
			
			fout.write(f"{fa_id}\t{prot_count}\t{json.dumps(fa_item)}\n")



def dump_sampled_fa_pros(sampled_fa_pros, sample_output_file):
	with open(sample_output_file, 'w') as fout:
		for fa_id in sampled_fa_pros:
			fa_item = sampled_fa_pros[fa_id]
			fout.write(f"{fa_id}\t{json.dumps(fa_item)}\n")



def sample_repres(fa_id, fa_info, 
	repre_pro_threshold, 
	repre_sample_rate, repre_sample_size_threshold,
	pro_sample_rate, pro_sample_size_threshold):

	# fa_info = {json.dumps(fa_info)}\n
	print(f'**************New starts for sampling representatives for a family domin.*********\n \
		Inputs: \n \
		fa_id = {fa_id} \n \
		repre_pro_threshold = {repre_pro_threshold},\n \
		repre_sample_rate = {repre_sample_rate}, \n\
		repre_sample_size_threshold = {repre_sample_size_threshold}, \n \
		pro_sample_rate = {pro_sample_rate}\n \
		pro_sample_size_threshold = {pro_sample_size_threshold}\n')

	sampled_repre_pros_dict = {}

	# sample from representatives which contains more than 10 proteins
	repre_gt_count = 0
	for repre_id in fa_info.keys():
		if repre_id != "family_id" and repre_id != "prot_count":
			pros_list = fa_info[repre_id]["prot_ids"]
			if len(pros_list) >= repre_pro_threshold:
				repre_gt_count += 1
	repre_sample_size = math.floor(repre_gt_count * repre_sample_rate)
	repre_sample_size = repre_sample_size \
			if repre_sample_size < repre_sample_size_threshold \
			else repre_sample_size_threshold
	
	# stop immediately
	if repre_sample_size <= 0:
		return sampled_repre_pros_dict

	nums = list(range(1, repre_gt_count))
	repre_sampled_nums = random.sample(nums, repre_sample_size)

	print(f'========== Assistant values:\n \
		repre_gt_count = {repre_gt_count}\n \
		repre_sample_size = {repre_sample_size}\n')
	

	repre_count = 0
	for repre_id in fa_info.keys():
		if repre_id != "family_id" and repre_id != "prot_count":
			pros_list = fa_info[repre_id]["prot_ids"]

			# Only large enough representatives will be sampled.
			if len(pros_list) >= repre_pro_threshold:
				repre_count += 1
				if repre_count in repre_sampled_nums:
					
					sampled_pros_list = []
					pro_sample_size = math.floor(len(pros_list) * pro_sample_rate)
					pro_sample_size = pro_sample_size \
						if pro_sample_size < pro_sample_size_threshold \
						else pro_sample_size_threshold

					if pro_sample_size <= 0:
						continue

					pro_sample_nums = random.sample(list(range(1, len(pros_list))), 
						pro_sample_size)
					print(f'^^^^^^^^^Assistant values for each representative:\n \
						pro_sample_size = {pro_sample_size}\n')
					
					pro_count = 0
					for pro_pair in pros_list:
						pro_count += 1
						if pro_count in pro_sample_nums:
							sampled_pros_list.append(pro_pair)

					sampled_repre_pros_dict[repre_id] = sampled_pros_list
	return sampled_repre_pros_dict



def sample_pros_by_family(sorted_fa_pros):
	threshold = 100

	repre_pro_threshold = 10
	family_sample_rate = 0.1

	repre_sample_rate = 0.3
	repre_sample_size_threshold = 4

	pro_sample_rate = 0.3
	pro_sample_size_threshold = 4

	count_gt = 0
	fa_pros_sample_size = 25
	sampled_fas = {}

	pros = sorted_fa_pros.items()

	for fa_id, fa_info in pros:
		if fa_info["prot_count"] > threshold:
			count_gt += 1

	target_sample_count = math.floor(count_gt * family_sample_rate)
	nums = list(range(1, count_gt))
	sampled_nums = random.sample(nums, fa_pros_sample_size)
	
	item_id = 0
	for fa_id, fa_info in pros:
		if fa_info["prot_count"] > threshold:
			item_id += 1
			if item_id in sampled_nums:
				sampled_family_pros = sample_repres(fa_id, fa_info, 
					repre_pro_threshold, 
					repre_sample_rate, repre_sample_size_threshold, 
					pro_sample_rate, pro_sample_size_threshold)
				if len(sampled_family_pros) > 0:
					sampled_fas[fa_id] = sampled_family_pros

	return sampled_fas




if __name__ == "__main__":
	random.seed(42)

	protein_file_name = "../../generated_data/scop-protein-whole-info.txt"
	count_file_name = "../../generated_data/scop-fa-stat.txt"
	sampled_fa_pros_file_name = "../../generated_data/scop-fa-sample.txt"
	
	sorted_fa_pros = count_family(protein_file_name)
	dump_fa_pros(sorted_fa_pros, count_file_name)

	sampled_fas = sample_pros_by_family(sorted_fa_pros)
	dump_sampled_fa_pros(sampled_fas, sampled_fa_pros_file_name)


