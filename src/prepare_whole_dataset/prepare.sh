#!/bin/bash

create_path() {
	given_file=$1
	given_file_dir=$(dirname "$given_file")
	if [ ! -d "$given_file_dir" ]; then
	    mkdir -p "$given_file_dir"
	    echo "Directory $given_file_dir created."
	fi
}


### 1. Uniq all the proteins' pdb_id
uniq_pdbs="../../generated_data/whole_pdbs/uniq_pdb_list.txt"
#create_path $uniq_pdbs

# python3 get_uniq_proteins.py $uniq_pdbs


### 2. Download all the pdb files for future use.
output_file="../../generated_data/whole_pdbs/sequences/protein_sequences"
err_file="../../generated_data/whole_pdbs/sequences.err/protein_sequences"
create_path $output_file
create_path $err_file

start=32001
end=40000
interval=1000
run_times=0

for ((i=start; i<=end; i+=interval)); do
	start_row=$i
	end_row=$((i + interval - 1))
	echo "start_row=$start_row, end_row=$end_row"
    python3 request_whole_pdb.py $uniq_pdbs $output_file $err_file $start_row $end_row
    sleep 100

    run_times=$((run_times + 1))
    if [ $((run_times % 10)) -eq 0 ]; then
    	sleep 300
    fi
done
