import requests
import json
import time
import sys
from Bio import PDB
from io import StringIO


def request_pdb(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id.lower()}.pdb'
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.text
    else:
        return f"******* Failed for pdb_id: {pdb_id} *******\n{response}\n"


# Parse the received PDB information.
def parse_pdb(pdb_info, chain_id):
    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)

    # Load the PDB file
    pdb_file = StringIO(pdb_info)
    structure = parser.get_structure('protein', pdb_file)

    # Initialize an empty sequence string
    sequence = ""

    # Iterate over models, chains, and residues to extract the sequence
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    # Check if the residue is an amino acid
                    if PDB.is_aa(residue):
                        # deprecated
                        # sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                        pro_name = residue.get_resname()
                        # unusual protein will be reflected as 'X'
                        if pro_name not in PDB.Polypeptide.protein_letters_3to1:
                            sequence += 'X'
                        else:
                            sequence += PDB.Polypeptide.protein_letters_3to1[pro_name]

    return sequence


def main(input_file, output_file, err_file, start_row, end_row):
    row_num = 0

    with open(input_file, 'r') as fin, \
        open(f"{output_file}.{start_row}.{end_row}.txt", 'w') as fout, \
        open(f"{err_file}.{start_row}.{end_row}.txt", 'w') as ferr:
        for line in fin:
            row_num += 1
            if row_num < start_row or row_num > end_row:
                print(f"The given row num {row_num} is not in range [{start_row}, {end_row}]")
                continue
            
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"Invalid input from {input_file} on line {row_num}.")
                continue
            
            pdb_id = parts[0]
            chain_ids = json.loads(parts[1])

            response_content = request_pdb(pdb_id)
            if "Failed for" not in response_content:
                for chain_id in chain_ids:
                    amino_acid_seq = parse_pdb(response_content, chain_id)
                    fout.write(f"{pdb_id}\t{chain_id}\t{amino_acid_seq}\n")
                    print(f"Row {row_num}: Protein Sequence obtained for PDB ID {pdb_id}, Chain ID {chain_id}.")
            else:
                ferr.write(f"Request pdb data failed for pdb_id: {pdb_id}, {response_content}\n")
                print(f"Failed to obtain protein sequence for {pdb_id}.")


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    err_file = sys.argv[3]
    start_row = int(sys.argv[4])
    end_row = int(sys.argv[5])

    main(input_file, output_file, err_file, start_row, end_row)
