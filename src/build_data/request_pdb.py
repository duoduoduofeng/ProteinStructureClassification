import requests
import json
import time
import sys
from Bio import PDB
from io import StringIO


# Request for the PDB file of a certain protein.
def get_protein_sequence(pdb_id, chain_id):
    url = f'https://files.rcsb.org/download/{pdb_id.lower()}.pdb'
    response = requests.get(url)
    
    if response.status_code == 200:
        amino_acid_seq = parse_pdb(response.text, chain_id)
        return f"{pdb_id}\t{chain_id}\t{amino_acid_seq}\n"
    else:
        return f"******* Failed for pdb_id: {pdb_id}, \
            chain_id: {chain_id} *******\n{response}\n"


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


# Load the sampled proteins.
def load_proteins(sample_file):
    sample_proteins_list = []
    with open(sample_file, 'r') as fin:
        row_num = 0
        for line in fin:
            row_num += 1
            fields = line.strip().split('\t')
            if len(fields) < 2:
                print(f"=== WARNING === The sample file contains \
                    illegal information on Row {row_num}")
                continue
            info = json.loads(fields[1])
            for pros in info.values():
                for pro in pros:
                    if pro not in sample_proteins_list:
                        sample_proteins_list.append(pro)
                    else:
                        print(f"=== WARNING === Duplicate protein {pro} \
                            on Row {row_num}")

    return sample_proteins_list


def toy_example_main():
    pdb_ids = ["6Q05", "5C23", "2QDG"]
    chain_ids = ["B", "A", "B"]

    for pdb_id, chain_id in zip(pdb_ids, chain_ids):
        protein_sequence = get_protein_sequence(pdb_id, chain_id)
        
        if protein_sequence:
            print(f"Protein Sequence for PDB ID {pdb_id}, \
                Chain ID {chain_id}:\n{protein_sequence}")
        else:
            print(f"No sequence found for PDB ID {pdb_id}, Chain ID {chain_id}.")


def main(input_file, output_file, err_file):
    sample_proteins_list = load_proteins(input_file)

    with open(output_file, 'w') as fout:
        with open(err_file, 'w') as ferr:
            for pdb_id, chain_id in sample_proteins_list:
                time.sleep(0.2)
                protein_sequence = get_protein_sequence(pdb_id, chain_id)
                
                if "Failed for pdb_id" not in protein_sequence:
                    fout.write(protein_sequence)
                    print(f"Protein Sequence obtained for PDB ID {pdb_id}, Chain ID {chain_id}.")
                else:
                    # retry
                    time.sleep(1)
                    protein_sequence_2 = get_protein_sequence(pdb_id, chain_id)
                    if "Failed for pdb_id" not in protein_sequence_2:
                        fout.write(protein_sequence_2)
                        print(f"************* After twice trials, protein Sequence obtained \
                            for PDB ID {pdb_id}, Chain ID {chain_id}.")
                    else:
                        ferr.write(protein_sequence_2)
                        print(f"Failed to obtain protein sequence for {pdb_id}\t{chain_id}")


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    err_file = sys.argv[3]

    main(input_file, output_file, err_file)
