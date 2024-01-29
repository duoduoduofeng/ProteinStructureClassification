#!/bin/bash
import json
import sys


def load_rs(rs_file):
	recall_dict = {}
	precision_dict = {}

	with open(rs_file, 'r') as fin:
		row_num = 0
		for line in fin:
			row_num += 1

			if row_num == 1:
				continue

			parts = line.strip().split("\t")
			real_distance = int(float(parts[4]))
			predict_distance = json.loads(parts[5])[0]
			discrete_predict_distance = -1

			# if real_distance == 0:
				# print(line)

			if predict_distance <= 0.5:
				discrete_predict_distance = 0
			elif predict_distance <= 1.8:
				discrete_predict_distance = 1
			elif predict_distance <= 3.5:
				discrete_predict_distance = 2
				# print(line)
			elif predict_distance <= 6:
				discrete_predict_distance = 4
			elif predict_distance <= 12:
				discrete_predict_distance = 8
			elif predict_distance <= 30:
				discrete_predict_distance = 16
			else:
				discrete_predict_distance = -1

			isright = 0
			if real_distance == discrete_predict_distance:
				isright = 1

			# recall_dict statistics
			if real_distance not in recall_dict:
				recall_dict[real_distance] = {}

			if isright == 1:
				if "right" not in recall_dict[real_distance]:
					recall_dict[real_distance]["right"] = 0
				recall_dict[real_distance]["right"] += 1
			else:
				if "wrong" not in recall_dict[real_distance]:
					recall_dict[real_distance]["wrong"] = 0
				recall_dict[real_distance]["wrong"] += 1

			# precision_dict statistics
			if discrete_predict_distance not in precision_dict:
				precision_dict[discrete_predict_distance] = {}

			if isright == 1:
				if "right" not in precision_dict[discrete_predict_distance]:
					precision_dict[discrete_predict_distance]["right"] = 0
				precision_dict[discrete_predict_distance]["right"] += 1
			else:
				if "wrong" not in precision_dict[discrete_predict_distance]:
					precision_dict[discrete_predict_distance]["wrong"] = 0
				precision_dict[discrete_predict_distance]["wrong"] += 1

	whole_right = 0
	whole_wrong = 0
	for va in recall_dict:
		if "right" not in recall_dict[va]:
			recall_dict[va]["right"] = 0
		if "wrong" not in recall_dict[va]:
			recall_dict[va]["wrong"] = 0
		
		recall_dict[va]["recall"] = float(recall_dict[va]["right"]/(recall_dict[va]["right"] + recall_dict[va]["wrong"]))
		recall_dict[va]["recall"] = round(recall_dict[va]["recall"], 2)
		
		whole_right += recall_dict[va]["right"]
		whole_wrong += recall_dict[va]["wrong"]
	
	recall_dict["whole_recall"] = float(whole_right / (whole_right + whole_wrong))
	recall_dict["whole_recall"] = round(recall_dict["whole_recall"], 2)

	for va in precision_dict:
		if "right" not in precision_dict[va]:
			precision_dict[va]["right"] = 0
		if "wrong" not in precision_dict[va]:
			precision_dict[va]["wrong"] = 0
		
		precision_dict[va]["precision"] = float(precision_dict[va]["right"]/(precision_dict[va]["right"] + precision_dict[va]["wrong"]))
		precision_dict[va]["precision"] = round(precision_dict[va]["precision"], 2)

		whole_right += precision_dict[va]["right"]
		whole_wrong += precision_dict[va]["wrong"]
	
	precision_dict["whole_precision"] = float(whole_right / (whole_right + whole_wrong))
	precision_dict["whole_precision"] = round(precision_dict["whole_precision"], 2)
	
	return recall_dict, precision_dict


if __name__ == "__main__":
	# rs_file = "../../generated_data/whole_pdbs/datasets/try_tp_3/result/predict_result.validate.txt.20240119_014953"
	rs_file = sys.argv[1]
	recall_dict, precision_dict = load_rs(rs_file)
	print(json.dumps(recall_dict))
	print(json.dumps(precision_dict))
