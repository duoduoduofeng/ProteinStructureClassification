#!/bin/bash

# mode="train"
mode="validate"

# 1. merge_full_fields.py
merged_data_file_name="../../generated_data/scop-protein-whole-info.txt"
# python3 merge_full_fields.py $merged_data_file_name


# 2. sample_by_family.py
protein_file_name="../../generated_data/scop-protein-whole-info.txt"
count_file_name="../../generated_data/$mode/scop-fa-stat.txt"
sampled_fa_pros_file_name="../../generated_data/$mode/scop-fa-sample.txt"
# theseed=42
# theseed=102
theseed=1018

python3 sample_by_family.py $protein_file_name $count_file_name $sampled_fa_pros_file_name $theseed


# 3. request_pdb.py
input_file="../../generated_data/$mode/scop-fa-sample.txt"
output_file="../../generated_data/$mode/protein_samples_seq.txt"
err_file="../../generated_data/$mode/protein_samples_err.txt"

python3 request_pdb.py $input_file $output_file $err_file


# 4. label_pair_distance.py
protein_info_file="../../generated_data/scop-protein-whole-info.txt"
sample_protein_file="../../generated_data/$mode/protein_samples_seq.txt"
whole_dataset_file="../../generated_data/$mode/sample_proteins_dataset.whole.txt"
model_dataset_file="../../generated_data/$mode/sample_proteins_dataset.txt"
validate_dataset_file="../../generated_data/$mode/sample_proteins_dataset.validate.txt"
theseed2=2023

python3 label_pair_distance.py $protein_info_file $sample_protein_file $whole_dataset_file $model_dataset_file $validate_dataset_file $theseed2
