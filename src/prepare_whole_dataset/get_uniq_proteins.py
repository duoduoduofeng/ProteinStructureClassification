import sys
import json


def uniq_proteins(proteins_file, output_file):
	# Dict runs much faster than list
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
					new_chains = []
					new_chains.append(chain_id)
					uniq_pros[pdb_id] = new_chains
				else:
					cur_chain = uniq_pros[pdb_id]
					if chain_id not in cur_chain:
						cur_chain.append(chain_id)

	print(f"There are {len(uniq_pros)} uniq pdb_id in total.\n")

	count_uniq_proteins = 0
	with open(output_file, 'w') as fout:
		for pdb_id in uniq_pros:
			count_uniq_proteins += len(uniq_pros[pdb_id])
			fout.write(f"{pdb_id}\t{json.dumps(uniq_pros[pdb_id])}\n")
		print(f"############ There are {count_uniq_proteins} uniq pdb_id, chain_id pairs. ############")


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

	merged_data_file_name = sys.argv[1]
	uniq_proteins(meta_info[0], merged_data_file_name)
