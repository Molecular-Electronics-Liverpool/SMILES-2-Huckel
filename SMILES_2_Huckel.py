from rdkit import Chem
import numpy as np
import os

def get_output_file_path():

    # Prompt for dir
    directory = input("Enter the full directory path to the folder you wish to save the output to: ").strip()

    # Check if dir exists and create if not
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"A pre-existing directory with the path: '{directory}' did not exist, so one was created at that location")

    # Prompt user for filename
    file_name_sansTXT = input("Enter a name for the Huckel file (e.g. metaBenzene, paraBenzene etc...): ").strip()
    file_name = file_name_sansTXT + ".txt"

    # Create the full output path
    output_file = os.path.join(directory, file_name)
    return output_file

def find_r_group_indices(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)  # Use sanitize=False to bypass default sanitization
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    r_indices = []
    for atom in mol.GetAtoms():
        # Check for wildcard atoms, [*] or [R]
        if atom.GetSymbol() == '*':
            #or atom.GetSmarts() == '[R]':
            r_indices.append(atom.GetIdx())
    
    return r_indices


def remove_atoms_from_smiles(smiles, atom_indices):
    # Parse the SMILES string into an RDKit molecule
    mol = Chem.MolFromSmiles(smiles, sanitize=False)  # Don't sanitize to avoid the valency error
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    # Create an editable molecule
    editable_mol = Chem.RWMol(mol)
    
    # Remove atoms specified by the list of indices
    for atom_idx in sorted(atom_indices, reverse=True):  # Sort in reverse to avoid index shift
        editable_mol.RemoveAtom(atom_idx)
    
    # Convert the modified molecule back to SMILES
    new_smiles = Chem.MolToSmiles(editable_mol)
    
    return new_smiles


def smiles_to_huckel_matrix(smiles, output_file):
    # Replace [R] with '*' for processing
    modified_smiles = smiles.replace('[R]', '*')  # Use '*' as a placeholder for R

    # Find and remove R groups in the modified SMILES
    r_group_indices = find_r_group_indices(modified_smiles)
    modified_smiles_no_r = remove_atoms_from_smiles(modified_smiles, r_group_indices)

    # Sanitize the modified SMILES (after removing R groups) to fix valency issues
    try:
        mol_for_error = Chem.MolFromSmiles(modified_smiles_no_r, sanitize=True)
    except Exception as e:
        print(f"Error during sanitization: {e}")
        return
    
    if mol_for_error is None:
        raise ValueError("Invalid SMILES string after removing R groups.")
    
    # Parse the modified SMILES string into an RDKit molecule
    mol = Chem.MolFromSmiles(modified_smiles, sanitize=False)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    # Step 1: Identify connection points
    connection_points = []
    original_index_map = {}
    connection_hoppings = []
    
    # Get atom indices and their symbols
    atom_indices = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_indices.append(atom.GetSymbol())
        original_index_map[idx] = atom.GetIdx()  # Store original index

    # Identify R groups and their neighbors
    for idx, symbol in enumerate(atom_indices):
        if symbol == '*':  # Check for the wildcard symbol
            # Add the index of the neighboring atom(s) connected to the R group
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                if neighbor.GetSymbol() != '*':  # Exclude R groups
                    connection_points.append(original_index_map[neighbor.GetIdx()])  # Store 1-based index
                    connection_hoppings.append(float(input(f"Enter the Resonance integral for Electrode-{neighbor.GetSymbol()} bond: ")))


    # Step 2: Create a new molecule excluding the R groups (*)
    new_mol = Chem.RWMol()  # Create a new editable molecule

    # Keep track of mapping from original indices to new indices
    index_map = {}

    # Add atoms to the new molecule
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() != '*':  # Only keep non-placeholder atoms
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())  # Preserve radical state
            new_idx = new_mol.AddAtom(new_atom)
            index_map[original_index_map[idx]] = new_idx  # Map original index to new index

    # Add bonds to the new molecule
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        if i in index_map and j in index_map:  # Only add bonds between kept atoms
            new_mol.AddBond(index_map[i], index_map[j], bond.GetBondType())
    
    # Finalize the new molecule
    new_mol = new_mol.GetMol()  # Convert to immutable molecule

    # Get the number of atoms in the new molecule
    num_atoms = new_mol.GetNumAtoms()
    
    # Initialize an empty matrix for the Huckel Hamiltonian
    matrix = np.full((num_atoms, num_atoms), "", dtype=object)
    
    # Dictionary to store unique atom symbols and bond types for user input
    coulomb_integrals = {}
    resonance_integrals = {}

    # Step 3: Collect unique atom types for Coulomb integrals, accounting for radicals
    for atom in new_mol.GetAtoms():
        element_symbol = atom.GetSymbol()
        is_radical = atom.GetNumRadicalElectrons() > 0
        
        # Key for radical vs. non-radical
        key = f"{element_symbol}_radical" if is_radical else element_symbol
        if key not in coulomb_integrals:
            integral_type = "(radical)" if is_radical else ""
            coulomb_integrals[key] = float(input(f"Enter the Coulomb integral for {element_symbol} {integral_type}: "))
    
    # Step 4: Collect unique bond types for Resonance integrals
    bond_types = {Chem.BondType.SINGLE: '-', Chem.BondType.DOUBLE: '=', Chem.BondType.TRIPLE: '#'}
    for bond in new_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Get bond type symbol
        bond_type_symbol = bond_types.get(bond.GetBondType(), '*')
        bond_symbol = f"{new_mol.GetAtomWithIdx(i).GetSymbol()}{bond_type_symbol}{new_mol.GetAtomWithIdx(j).GetSymbol()}"
        
        # Prompt for resonance integral if bond type hasn't been encountered
        if bond_symbol not in resonance_integrals:
            resonance_integrals[bond_symbol] = float(input(f"Enter the Resonance integral for {bond_symbol} bond: "))
    
    # Step 5: Fill the matrix with the Coulomb and Resonance integrals
    for i, atom in enumerate(new_mol.GetAtoms()):
        element_symbol = atom.GetSymbol()
        is_radical = atom.GetNumRadicalElectrons() > 0
        key = f"{element_symbol}_radical" if is_radical else element_symbol
        matrix[i, i] = coulomb_integrals[key]  # Diagonal with Coulomb integrals
    
    for bond in new_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Get bond symbol
        bond_type_symbol = bond_types.get(bond.GetBondType(), '*')
        bond_symbol = f"{new_mol.GetAtomWithIdx(i).GetSymbol()}{bond_type_symbol}{new_mol.GetAtomWithIdx(j).GetSymbol()}"
        
        # Check if bond symbol exists in resonance integrals
        if bond_symbol in resonance_integrals:
            matrix[i, j] = resonance_integrals[bond_symbol]
            matrix[j, i] = resonance_integrals[bond_symbol]     # Ensure symmetric matrix
    
    # Write the connection points and matrix to a text file
    with open(output_file, 'w') as f:
        f.write('\t'.join(map(str, connection_hoppings)))       # Write connection point hoppings
        f.write('\n' + '\t'.join(map(str, connection_points)))  # Write connection points
        for row in matrix:
            f.write('\n' + '\t'.join(map(str, row)))

    print(f"Huckel matrix with integrals written to {output_file}")

# Main script execution
output_file = get_output_file_path()
smiles_input = input("Enter a SMILES string: ")
smiles_to_huckel_matrix(smiles_input, output_file)
















