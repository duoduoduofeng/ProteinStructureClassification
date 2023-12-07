import requests
from Bio import PDB
from io import StringIO

def get_protein_sequence(pdb_id, chain_id):
    url = f'https://files.rcsb.org/download/{pdb_id.lower()}.pdb'
    response = requests.get(url)
    
    if response.status_code == 200:
        amino_acid_seq = parse_pdb(response.text, chain_id)
        return f"{pdb_id}\t{chain_id}\t{amino_acid_seq}"
    else:
        return f"******* Failed for pdb_id: {pdb_id}, \
            chain_id: {chain_id} *******\n{response}\n"



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
                        # sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                        sequence += PDB.Polypeptide.protein_letters_3to1[residue.get_resname()]

    return sequence



if __name__ == "__main__":
    pdb_ids = ["6Q05", "5C23", "2QDG"]
    chain_ids = ["B", "A", "B"]

    for pdb_id, chain_id in zip(pdb_ids, chain_ids):
        protein_sequence = get_protein_sequence(pdb_id, chain_id)
        
        if protein_sequence:
            print(f"Protein Sequence for PDB ID {pdb_id}, Chain ID {chain_id}:\n{protein_sequence}")
        else:
            print(f"No sequence found for PDB ID {pdb_id}, Chain ID {chain_id}.")
