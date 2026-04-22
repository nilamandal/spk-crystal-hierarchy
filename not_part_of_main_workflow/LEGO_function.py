import networkx as nx
import os
#from pylada.crystal import read, neighbors, write, supercell
#import pylada.periodic_table as PT
import numpy as np
import spglib
from copy import deepcopy
from itertools import product
#from vladan.format_spglib import *
from numpy.linalg import norm
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.dimensionality import get_structure_components
from pymatgen.core import Structure
from pymatgen.core import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# Covalent radii and tolerance values
covalent_radii = {
    'P': 1.11, 'As': 1.21, 'Sb': 1.40, 'Bi': 1.60, 'Si': 1.16, 'Ge': 1.21, 'Sn': 1.40, 'Pb': 1.44,
    'Sr': 1.85, 'Cs': 2.32, 'K': 1.96, 'Na': 1.60, 'Al': 1.26, 'Ga': 1.24, 'In': 1.42, 'Tl': 1.44,
    'Zn': 1.20, 'Cd': 1.44, 'Hg': 1.44, 'Mn': 1.19
}

electronegativity_lookup= {'H':2.2, 'Li':0.98, 'Be':1.57, 'Na':0.93, 'Mg':1.31, 'Al':1.61, 'Si':1.9, 'P':2.19, 'K':0.82,
    'Ca':1, 'Mn':1.55, 'Zn':1.65, 'Ga':1.81, 'Ge':2.01, 'As':2.18, 'Rb':0.82, 'Sr':0.95, 'Cd':1.69, 'In':1.69, 'Sn':1.96, 'Sb':2.05,
    'Cs':0.79, 'Ba':0.89, 'Yb':1.1, 'Hg':2, 'Tl':1.62, 'Pb':2.33, 'Bi':2.02}

#Step 1: find cations and anions

def get_atom_types_counts(structure):

    """
    Function to get counts of each atom type in a structure.

    Args:
        - structure: Pylada structure object.

    Returns:
        - atom_types_counts: Dictionary containing counts of each atom type.
    """
    atom_types_counts = {}
    for site in structure:
        #print(site)
        atom_type = str(site.specie)
        atom_types_counts[atom_type] = atom_types_counts.get(atom_type, 0) + 1
    return atom_types_counts


def calculate_difference(structure, en_tol=0.65):
    """
    Function to group cations and anions.
    It also calculates the difference in electronegativity between cations and anions in a structure.

    Args:
        - structure: Pylada structure object.
        - en_tol (float, optional): The tolerance for considering atoms as cations or anions based on electronegativity.
          Defaults to 0.65.

    Returns:
        - tuple: A tuple containing the following:
               - A dictionary containing counts of cations, where the keys are atom types.
               - A dictionary containing counts of anions, where the keys are atom types.
               - The total number of cations.
               - The total number of anions.
    """
    atom_types_counts = get_atom_types_counts(structure)
    #print(atom_types_counts)
    cations = {}
    anions = {}

    # Identify pnictogens (group 15 elements) as anions
    pnictogens = {"N", "P", "As", "Sb", "Bi"}
    for atom_type in atom_types_counts:
        if atom_type in pnictogens:
            anions[atom_type] = atom_types_counts[atom_type]

    Ln = {"Yb", "Eu"}

    for atom_type in atom_types_counts:
        if atom_type in Ln:
            cations[atom_type] = atom_types_counts[atom_type]

    # Handle the second element as a cation by default if there are only two types of atoms
    remaining_atom_types = set(atom_types_counts.keys()) - set(anions.keys()) - set(cations.keys())
    if len(atom_types_counts) == 2 and remaining_atom_types:
        second_atom_type = remaining_atom_types.pop()
        cations[second_atom_type] = atom_types_counts[second_atom_type]
    else:
        # For more than two elements, apply the EN criteria
        max_en_atom_type = max(anions, key=lambda x: electronegativity_lookup[x])
        for atom_type in remaining_atom_types:
            en_difference = abs(electronegativity_lookup[atom_type] - electronegativity_lookup[max_en_atom_type])
            if en_difference > en_tol:
                cations[atom_type] = atom_types_counts[atom_type]
            else:
                anions[atom_type] = atom_types_counts[atom_type]

    # Calculate the total number of cations and anions
    total_cations = sum(cations.values())
    total_anions = sum(anions.values())

    return cations, anions, total_cations, total_anions

#Step 2: Select the LEGO by combining the anionic framework

def process_structure(path, en_tol=0.65, covalent_radii_tolerance=0.3):
    """Processes a crystal structure to identify and return connected components based on covalent bonds."""
    structure = Structure.from_file(path)
    atom_types_counts = get_atom_types_counts(structure)
    _, anions, _, _ = calculate_difference(structure, en_tol)
    print(f"Processing structure: {path}")

    G = nx.Graph()
    processed_bonds = set()

    target_anions = ['P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'Al', 'Ga', 'In', 'Tl']
    present_anions = [anion for anion in target_anions if anion in atom_types_counts]

    coordination_dict = {anion: [0] * len(structure) for anion in present_anions}

    atom_coordination = {}

    for i in range(len(structure)):
        G.add_node(i)
        structure_i_type= str(structure[i].specie)
        #print(structure[])
        if structure_i_type in anions:
            ngh= structure.get_all_neighbors(r=8,sites=[structure[i]],numerical_tol=0.3)[0]
            print(ngh)

            #print(len(ngh))
            #ngh = neighbors(structure, 2, structure[i], 0.3)
            for j in range(len(ngh)):
                #bond_length = ngh[j][-1]
                bond_length= structure[i].distance(ngh[j])

                for k in range(len(structure)):
                    structure_k_type= str(structure[k].specie)
                    if structure_k_type in anions:
                        radii_1 = covalent_radii[structure_i_type] + covalent_radii_tolerance
                        radii_2 = covalent_radii[structure_k_type] + covalent_radii_tolerance
                        assert radii_1 is not None, f"Covalent radius not found for atom type: {structure_i_type} in covalent_radii dictionary"
                        assert radii_2 is not None, f"Covalent radius not found for atom type: {structure_k_type} in covalent_radii dictionary"
                        radii_total=radii_1+radii_2

                        if i < k and ngh[j] == structure[k] and bond_length <= radii_total:

                            G.add_edge(i, k)
                            print(f"Added edge between {i} and {k} (Bond length: {bond_length})")

                            if structure_i_type == structure_k_type:
                                if structure_i_type in coordination_dict:
                                    coordination_dict[structure_i_type][i] += 1
                                    coordination_dict[structure_k_type][k] += 1

                        else:
                            print(f"Rejected bond between {i} and {k}: Bond length {bond_length} >= {radii_1 + radii_2}")

    connected_components = list(nx.connected_components(G))
    filtered_components = [comp for comp in connected_components if len(comp) > 1]

    for idx, component in enumerate(filtered_components):
        print(f"Group {idx + 1} for structure {path}: {component}")

    # Corrected loop for assigning coordination numbers
    for anion in present_anions:
        for i in range(len(structure)):
            structure_i_type= str(structure[i].specie)
            if structure_i_type == anion:
                for j in range(len(structure)):
                    structure_j_type= str(structure[j].specie)
                    if i != j and structure_j_type == anion:  # Only consider bonds between same anion types
                    # The coordination number is directly the value from the coordination_dict
                        atom_coordination[i] = coordination_dict[anion][i]

    print(f"Atom Coordination: {atom_coordination}")

    return filtered_components, structure, atom_coordination


#Step 3: Calculate charge of LEGO

def cartesian_to_direct(cart_coords, cell):
    """Convert Cartesian coordinates to Direct coordinates."""
    cell = np.array(cell)
    cell_inv = np.linalg.inv(cell)
    return np.dot(cart_coords, cell_inv.T)

def direct_to_cartesian(direct_coords, cell):
    """Convert Direct coordinates to Cartesian coordinates."""
    return np.dot(direct_coords, cell)

def apply_pbc(fractional_vector):
    # Normalize the vector to ensure all components are in the range [0, 1)
    normalized_vector = (fractional_vector % 1)
    return normalized_vector

def get_wyckoff_info(structure, symprec=0.1):
    # Get the symmetry dataset
    sym_dataset = spglib.get_symmetry_dataset(to_spglib(structure), symprec=symprec)
    wyckoff_positions = sym_dataset['wyckoffs']

    wyckoff_info = []
    for i, atom in enumerate(structure):
        atom_type = atom.type
        wyckoff_position = wyckoff_positions[i]
        wyckoff_info.append((i, atom_type, wyckoff_position))
    return wyckoff_info

def assign_charges_and_find_neutral_combination(path):
    _, _, atom_coordination = process_structure(path)
    structure = read.poscar(path + 'CONTCAR')

    # Define group elements
    group_1 = ['Li', 'Na', 'K', 'Rb', 'Cs']
    group_2 = ['Be', 'Mg', 'Ca', 'Sr', 'Ba']
    group_12 = ['Zn', 'Cd', 'Hg','Eu', 'Yb']
    group_Mn = ['Mn']
    group_13 = ['B', 'Al', 'Ga', 'In', 'Tl']
    group_14 = ['C', 'Si', 'Ge', 'Sn', 'Pb']
    group_15 = ['P', 'As', 'Sb', 'Bi']

    Charges = {}

    # Assign charges based on coordination number and element type
    # Assign charges for group 15 elements based on CN
    for i, site in enumerate(structure):
        element = site.type
        if element in group_15:
            CN = atom_coordination.get(i, None)
            Charges[i] = {0: -3, 1: -2, 2: -1, 3: 0, 4: +1}.get(CN, 0)

    # Assign charges for group 1, 2, and 12 elements
    for i in range(len(structure)):
        element = structure[i].type
        if i in Charges:
            continue

        if element in group_1:
            Charges[i] = +1
        elif element in group_2 or element in group_12:
            Charges[i] = +2

    group_Mn_indices = [i for i in range(len(structure)) if structure[i].type in group_Mn]

    all_combinations_group_Mn = []
    if group_Mn_indices:
        # Create all combinations of +2 and +3 for the number of Mn indices
        charge_combinations = product((+2, +3), repeat=len(group_Mn_indices))

        for charges in charge_combinations:
            combined_charges = {}  # Create a new dictionary for each combination
            for idx, charge in zip(group_Mn_indices, charges):
                combined_charges[idx] = charge  # Assign the charge to the respective index
            all_combinations_group_Mn.append(combined_charges.copy())  # Append this combination to the list

    all_combinations_group_13 = []
    all_combinations_group_14 = []

    # Attempt to assign charges for group 13

    group_13_indices = [i for i in range(len(structure)) if structure[i].type in group_13]

    if group_13_indices:
        # Separate indices based on their CN
        cn_1_indices = [i for i in group_13_indices if atom_coordination.get(i) == 1]
        cn_2_indices = [i for i in group_13_indices if atom_coordination.get(i) == 2]
        cn_3_indices = [i for i in group_13_indices if atom_coordination.get(i) == 3]
        cn_zero_indices = [i for i in group_13_indices if atom_coordination.get(i) in (None, 0)]

        # Define charge combinations for different CN values
        CN_1_combinations = [(+2,), (+0,)]
        CN_2_combinations = [(+1,)]
        CN_3_combinations = [(0,)]
        zero_combinations = [(+3,), (+1,)]

        # Generate all possible combinations of charges
        for CN_1_comb, CN_2_comb, CN_3_comb, zero_comb in product(CN_1_combinations, CN_2_combinations, CN_3_combinations,zero_combinations):

            combined_charges = {}

            # Assign charges for non-zero CN atoms
            for i in cn_1_indices:
                combined_charges[i] = CN_1_comb[0]

            for i in cn_2_indices:
                combined_charges[i] = CN_2_comb[0]

            for i in cn_3_indices:
                combined_charges[i] = CN_3_comb[0]

            # Assign charges for CN=0 atoms
            for i in cn_zero_indices:
                combined_charges[i] = zero_comb[0]

            # Append this full charge combination to all_combinations_group_14
            all_combinations_group_13.append(combined_charges.copy())


    # Now `all_combinations_group_13` should contain all the required combinations
    print("Final Combined Charges Group 13:", all_combinations_group_13)

    # Attempt to assign charges for group 14
    group_14_indices = [i for i in range(len(structure)) if structure[i].type in group_14]

    if group_14_indices:
        # Separate indices based on their CN
        cn_1_indices = [i for i in group_14_indices if atom_coordination.get(i) == 1]
        cn_2_indices = [i for i in group_14_indices if atom_coordination.get(i) == 2]
        cn_3_indices = [i for i in group_14_indices if atom_coordination.get(i) == 3]
        cn_4_indices = [i for i in group_14_indices if atom_coordination.get(i) == 4]
        cn_zero_indices = [i for i in group_14_indices if atom_coordination.get(i) in (None, 0)]

        # Define charge combinations for different CN values
        CN_1_combinations = [(+3,), (+1,)]
        CN_2_combinations = [(+2,), (0,)]
        CN_3_combinations = [(+1,)]
        CN_4_combinations = [(0,)]
        zero_combinations = [(+4,), (+2,)]

        # Generate all possible combinations of charges
        for CN_1_comb, CN_2_comb, CN_3_comb, CN_4_comb, zero_comb in product(CN_1_combinations, CN_2_combinations, CN_3_combinations, CN_4_combinations, zero_combinations):

            combined_charges = {}

            # Assign charges for non-zero CN atoms
            for i in cn_1_indices:
                combined_charges[i] = CN_1_comb[0]

            for i in cn_2_indices:
                combined_charges[i] = CN_2_comb[0]

            for i in cn_3_indices:
                combined_charges[i] = CN_3_comb[0]

            for i in cn_4_indices:
                combined_charges[i] = CN_4_comb[0]

            # Assign charges for CN=0 atoms
            for i in cn_zero_indices:
                combined_charges[i] = zero_comb[0]

            # Append this full charge combination to all_combinations_group_14
            all_combinations_group_14.append(combined_charges.copy())


    # Now `all_combinations_group_14` should contain all the required combinations
    print("Final Combined Charges Group 14:", all_combinations_group_14)


    # Check for charge neutrality with all combinations
    found_neutral_combination = False

    # Try all combinations from both groups independently or together
    if group_Mn_indices:
        for comb_Mn in all_combinations_group_Mn:
            # Deep copy of Charges for testing
            test_charges = deepcopy(Charges)  # Start with already assigned charges
            test_charges.update(comb_Mn)  # Add Mn charges

            if all_combinations_group_13:  # Check if there are any group 13 combinations
                for comb_13 in all_combinations_group_13:
                    #Deep copy test_charges to ensure it doesn't overwrite the previous combination
                    temp_charges = deepcopy(test_charges)
                    temp_charges.update(comb_13)  # Add group 13 charges

                    if all_combinations_group_14:  # Check if there are any group 14 combinations
                        for comb_14 in all_combinations_group_14:
                            # Deep copy temp_charges to ensure neutrality is checked for the current combination
                            final_charges = deepcopy(temp_charges)
                            final_charges.update(comb_14)  # Add group 14 charges

                            # Check for neutrality
                            if sum(final_charges.values()) == 0:
                                Charges.update(final_charges)  # Update Charges with the neutral combination
                                found_neutral_combination = True
                                break  # Stop after finding the first neutral combination

                    else:  # No group 14 combinations, check neutrality with group 13 and Mn
                        if sum(temp_charges.values()) == 0:
                            Charges.update(temp_charges)  # Update Charges with the neutral combination
                            found_neutral_combination = True
                            break  # Stop after finding the first neutral combination

                    if found_neutral_combination:
                        break  # Stop if neutral combination is found

            else:  # No group 13 combinations, check neutrality with Mn and group 14
                if all_combinations_group_14:
                    for comb_14 in all_combinations_group_14:
                        # Deep copy test_charges for group 14
                        temp_charges = deepcopy(test_charges)
                        temp_charges.update(comb_14)  # Add group 14
                        # Check for neutrality
                        if sum(temp_charges.values()) == 0:
                            Charges.update(temp_charges)  # Update Charges with the neutral combination
                            found_neutral_combination = True
                            break  # Stop after finding the first neutral combination

                else:  # No group 13 or group 14 combinations, check neutrality with just Mn
                    if sum(test_charges.values()) == 0:
                        Charges.update(test_charges)  # Update Charges with the neutral combination
                        found_neutral_combination = True
                        break  # Stop after finding the first neutral combination

            if found_neutral_combination:
                break  # Stop if neutral combination is found


    if all_combinations_group_13:
        for comb_13 in all_combinations_group_13:
            # Deep copy of Charges for testing
            test_charges = Charges.copy()  # Start with already assigned charges
            test_charges.update(comb_13)  # Add group 13 charges

            if all_combinations_group_14:  # Check if there are any group 14 combinations
                # Check neutrality with all group 14 combinations
                for comb_14 in all_combinations_group_14:
                    # Update test_charges with group 14 charges
                    test_charges.update(comb_14)

                    # Check for neutrality
                    if sum(test_charges.values()) == 0:
                        Charges.update(test_charges)  # Update Charges with the neutral combination
                        found_neutral_combination = True
                        break  # Stop after finding the first neutral combination

            else:  # No group 14 combinations, check neutrality with just group 13
                if sum(test_charges.values()) == 0:
                    Charges.update(test_charges)  # Update Charges with the neutral combination
                    found_neutral_combination = True
                    break  # Stop after finding the first neutral combination

            if found_neutral_combination:
                break  # Stop if neutral combination is found


    # Check if only group 14 elements exist
    if not group_13_indices and not group_Mn_indices and group_14_indices:
        for comb_14 in all_combinations_group_14:
            test_charges = Charges.copy()  # Start with already assigned charges
            test_charges.update(comb_14)  # Add group 14 charges

            # Check for neutrality
            if sum(test_charges.values()) == 0:
                Charges.update(test_charges)  # Update Charges with the neutral combination
                found_neutral_combination = True
                break  # Stop after finding the first neutral combination

    if sum(Charges.values()) == 0:
        print("Neutral structure achieved with initial charges.")
        return Charges
    else:
        print("Applying Wyckoff-based charge assignment...")

    # Define charge combinations
    group_combinations = {
        'group_1': [+1],
        'group_2_and_12': [+2],
        'group_Mn': [+3,+2],
        'group_13': [+3,+2,+1, 0],
        'group_14': [+4, +3, +2, +1, 0],
        'group_15': [0, -1, -2, -3]
    }

    wyckoff_data = get_wyckoff_info(structure)
    print("Wyckoff Data:", wyckoff_data)

    def apply_charge_combinations(group_indices, charge_combinations, charges):
        test_combinations = []

        # Group all unique Wyckoff positions together regardless of their specific site
        unique_wyckoffs = {}
        for idx, element, wyckoff in group_indices:
            wyckoff_key = (element, wyckoff)  # Combine element and Wyckoff as key

            # Allow any Wyckoff position to have any charge in charge_combinations
            unique_wyckoffs[wyckoff_key] = charge_combinations

        # Prepare Cartesian product of charges for each unique Wyckoff site
        wyckoff_keys = list(unique_wyckoffs.keys())
        charge_product = list(product(*[charge_combinations for _ in wyckoff_keys]))

        # Iterate over all possible combinations of charges for the Wyckoff sites
        for charges_list in charge_product:
            charges_copy = charges.copy()

            # Assign the current charge combination to all atoms in Wyckoff sites
            for (wyckoff_key, charge) in zip(wyckoff_keys, charges_list):
                for idx, element, wyckoff in group_indices:
                    if (element, wyckoff) == wyckoff_key:
                        charges_copy[idx] = charge

            test_combinations.append(charges_copy)

        return test_combinations


    def find_neutral_combinations(current_charges):
        all_combinations = [current_charges]

        # Define groups
        groups = {
        'group_1': [(i, structure[i].type, wyckoff) for i, atom_type, wyckoff in wyckoff_data if atom_type in group_1],
        'group_2_and_12': [(i, structure[i].type, wyckoff) for i, atom_type, wyckoff in wyckoff_data if atom_type in group_2 or atom_type in group_12],
        'group_Mn': [(i, structure[i].type, wyckoff) for i, atom_type, wyckoff in wyckoff_data if atom_type in group_Mn],
        'group_13': [(i, structure[i].type, wyckoff) for i, atom_type, wyckoff in wyckoff_data if atom_type in group_13],
        'group_14': [(i, structure[i].type, wyckoff) for i, atom_type, wyckoff in wyckoff_data if atom_type in group_14],
        'group_15': [(i, structure[i].type, wyckoff) for i, atom_type, wyckoff in wyckoff_data if atom_type in group_15]
        }

        # Reapply charges for each group
        for group_name, group_indices in groups.items():
            if group_indices:
                new_combinations = []
                for charge_set in all_combinations:
                    new_combinations.extend(apply_charge_combinations(group_indices, group_combinations[group_name], charge_set))
                all_combinations = new_combinations

        # Filter only neutral charge combinations
        neutral_combinations = [combo for combo in all_combinations if sum(combo.values()) == 0]
        filtered_combinations = []

        # Step 1: If there's exactly one neutral combination, return it immediately
        if len(neutral_combinations) == 1:
            filtered_combinations.append(neutral_combinations[0])
            print("Only one neutral combination found, returning it.")
            return filtered_combinations


        # Step 2: If no valid combination is found, try filtering for groups 13, 14, and 15
        if not filtered_combinations:
            print("Let's filter combinations.")
            for combo in neutral_combinations:
                valid = True

                # Step 1: Check group 15 elements (no more than 3 unique oxidation states)
                for idx, charge in combo.items():
                    element = structure[idx].type
                    if element in group_15:
                        element_charges = [combo[i] for i in range(len(structure)) if structure[i].type == element]
                        if len(set(element_charges)) > 2:
                            valid = False
                            break  # Stop if group 15 condition fails

                # Step 2: If group 15 check passed, proceed to check group 13/14 elements
                if valid:
                    group_13_14_checked = set()  # Track already checked elements to avoid redundancy
                    for idx, charge in combo.items():
                        element = structure[idx].type
                        if element in group_13 or element in group_14:
                            if element not in group_13_14_checked:
                                element_charges = [combo[i] for i in range(len(structure)) if structure[i].type == element]
                                if len(set(element_charges)) > 2:
                                    valid = False
                                    break  # Stop if group 13/14 condition fails
                                group_13_14_checked.add(element)

                # If valid combination is found  append
                if valid:
                    print(f"Relaxed filtered combination found: {combo}")
                    filtered_combinations.append(combo)

        # Step 3: If no valid combination is found even after relaxed filtering, return neutral combinations
        if not filtered_combinations:
            print("Even relaxed filtering yielded no results. Returning original neutral combinations.")
            filtered_combinations = neutral_combinations

        # Return the filtered combinations
        return filtered_combinations


    # Process the returned combination as unique
    neutral_combinations = find_neutral_combinations(Charges)
    if not neutral_combinations:
        raise ValueError("No combination of charges results in a neutral structure.")

    # Remove duplicate combinations and return one unique combination
    unique_combinations = []
    seen = set()
    for combo in neutral_combinations:
        sorted_items = tuple(sorted(combo.items()))
        if sorted_items not in seen:
            unique_combinations.append(combo)
            seen.add(sorted_items)

    if unique_combinations:
        print(f"Unique Combination: {unique_combinations}")
        return unique_combinations
    else:
        raise ValueError("No unique combination found.")

#Step 4: Find dimensionality of LEGO

def get_filtered_structures(path):

    filtered_components, _, _ = process_structure(path)
    filtered_structures = []
    structure = Poscar.from_file(path + 'CONTCAR').structure

    for idx, component in enumerate(filtered_components):


        # Deepcopy the original structure to create a new structure for each component
        sub_structure = deepcopy(structure)

        # Find indices of atoms not in the current component and remove them
        indices_to_remove = [i for i in range(len(sub_structure)) if i not in component]
        sub_structure.remove_sites(indices_to_remove)

        # Append the filtered substructure to the list
        filtered_structures.append(sub_structure)

        poscar = Poscar(sub_structure)
        poscar.write_file(f"POSCAR_filtered_{idx + 1}")

    return filtered_structures

def analyze_dimensionality_direction(path):

    filtered_structures = get_filtered_structures(path)
    dimension_info = []

    for i, structure in enumerate(filtered_structures):
        # Generate bonded structure using CrystalNN
        bonded_structure = CrystalNN(distance_cutoffs=(0.5, 0.5), x_diff_weight=0, porous_adjustment=False).get_bonded_structure(structure)

        dimension = get_dimensionality_larsen(bonded_structure)

        # Get components information
        structure_components = get_structure_components(
            bonded_structure,
            inc_orientation=True,
            inc_site_ids=False,
            inc_molecule_graph=False
        )

        orientations = []
        for component in structure_components:
            if "orientation" in component:
                orientations.append(component["orientation"])

        # Store only the first orientation or summarize orientations as needed
        orientation = orientations[0] if orientations else None

        dimension_info.append({
            "filtered_structure_index": i,
            "structure": structure,
            "dimensionality": dimension,
            "orientation": orientation,

        })
    return dimension_info

# Step 5: Consider 2D LEGOs and try to match the orientation if possible

def rotation_matrix_vectors(v1, v2):
    """Get the rotation matrix that rotates v1 onto v2 using
    Rodrigues' rotation formula.

    See more: https://math.stackexchange.com/a/476311

    Args:
        v1: initial vector
        v2: target vector

    Returns:
        3x3 rotation matrix
    """
    if np.allclose(v1, v2):
        # same direction
        return np.eye(3)

    if np.allclose(v1, -v2):
        # opposite direction: return a rotation of pi around the y-axis
        return np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    v = np.cross(v1, v2)
    norm = np.linalg.norm(v)
    c = np.vdot(v1, v2)

    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1.0 - c) / (norm * norm))

    # Validate the rotation matrix (R.T @ R = I)
    identity_check = np.allclose(np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), atol=1e-6)
    if not identity_check:
        raise ValueError("Generated rotation matrix is not orthogonal.")

    return rotation_matrix

def get_normal_vector(structure, miller_indices):
    """
    Calculates the normal vector of a plane with given Miller indices for a crystal structure.

    Handles both three-index (h, k, l) and four-index (h, k, i, l) notation for hexagonal systems.

    Args:
        structure (Structure): A Pymatgen Structure object.
        miller_indices (tuple): A tuple of three or four integers representing the Miller indices.

    Returns:
        np.array: The unit normal vector as a NumPy array.
    """
    # Convert 4-index (h, k, i, l) to 3-index (h, k, l) if necessary
    if len(miller_indices) == 4:
        h, k, i, l = miller_indices
        if not np.isclose(i, -(h + k)):
            raise ValueError("For hexagonal systems, i must equal -(h + k).")
    elif len(miller_indices) == 3:
        h, k, l = miller_indices
    else:
        raise ValueError("Miller indices must be a tuple of 3 or 4 integers.")

    # Get reciprocal lattice in Cartesian coordinates
    reciprocal_lattice = structure.lattice.reciprocal_lattice_crystallographic

    # Compute the normal vector in reciprocal space
    normal_vector = h * reciprocal_lattice.matrix[0] + \
                    k * reciprocal_lattice.matrix[1] + \
                    l * reciprocal_lattice.matrix[2]

    # Normalize the normal vector to make it a unit vector
    unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return unit_normal_vector

def update_structure_with_rotated_coords(original_structure, rotated_coords):
    """
    Creates a new Structure with rotated fractional coordinates.

    Parameters:
        original_structure (Structure): Original pymatgen Structure object.
        rotated_coords (ndarray): New fractional coordinates (Nx3 array).

    Returns:
        Structure: New Structure with updated coordinates.
    """
    new_structure = Structure(
        original_structure.lattice,
        original_structure.species,
        rotated_coords,
        coords_are_cartesian=False
    )
    return new_structure

def has_overlapping_atoms(structure, tol=0.5):
    """
    Checks whether the structure has overlapping atoms.
    It uses pymatgen's get_distance method between pairs of sites.

    Args:
        structure (Structure): A pymatgen Structure object.
        tol (float): Distance threshold below which atoms are considered overlapping.

    Returns:
        bool: True if any two atoms are closer than tol, else False.
    """
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            if structure.get_distance(i, j) < tol:
                return True
    return False

def match_orientation_std(structure1_path, structure2_path):
    """
    Performs all possible group replacements between two structures in Direct coordinates,
    standardizing each structure to a conventional cell before alignment.
    If an aligned structure has overlapping atoms (below a tolerance of 0.5 Å),
    the function returns None.
    """
    # Analyze dimensionality and direction for each structure
    dimension_info_1 = analyze_dimensionality_direction(structure1_path)
    dimension_info_2 = analyze_dimensionality_direction(structure2_path)

    aligned_structures = []

    # Loop over all combinations of dimension_info from both structures
    for group1 in dimension_info_1:
        for group2 in dimension_info_2:
            # Process only if both groups are 2D
            if group1["dimensionality"] == group2["dimensionality"] == 2:
                # Get the original structures from analysis
                struct1 = group1["structure"]
                struct2 = group2["structure"]

                # Standardize the cells using pymatgen's SpacegroupAnalyzer.
                # Using the conventional standard cell.
                struct1_std = SpacegroupAnalyzer(struct1, symprec=1.0e-8).get_conventional_standard_structure()
                struct2_std = SpacegroupAnalyzer(struct2, symprec=1.0e-8).get_conventional_standard_structure()

                miller1 = group1["orientation"]
                miller2 = group2["orientation"]

                # Convert Miller indices to Cartesian orientation vectors
                normal_vector1 = get_normal_vector(struct1_std, miller1)
                normal_vector2 = get_normal_vector(struct2_std, miller2)

                # Check if the orientations already match within tolerance
                if np.allclose(normal_vector1, normal_vector2, atol=0.1):
                    modified_structure2 = copy.deepcopy(struct2_std)
                    aligned_structures.append({
                        "aligned_structure": modified_structure2,
                        "group1_index": group1["filtered_structure_index"],
                        "group2_index": group2["filtered_structure_index"]
                    })
                else:
                    # Calculate the rotation matrix to align normal_vector2 with normal_vector1
                    rotation_matrix = rotation_matrix_vectors(normal_vector2, normal_vector1)
                    print("Rotation matrix:\n", rotation_matrix)

                    # Rotate lattice vectors
                    rotated_lattice_vectors = np.dot(struct2_std.lattice.matrix, rotation_matrix.T)
                    rotated_lattice = Lattice(rotated_lattice_vectors)

                    # Compare lattice parameters before and after rotation
                    original_lattice = struct2_std.lattice
                    original_lengths = original_lattice.abc
                    original_angles = original_lattice.angles

                    rotated_lengths = rotated_lattice.abc
                    rotated_angles = rotated_lattice.angles

                    print("Lattice parameters comparison before and after rotation:")
                    for i, (orig_len, rot_len) in enumerate(zip(original_lengths, rotated_lengths), 1):
                        print(f"  Length {chr(96 + i)}: Original = {orig_len:.3f}, Rotated = {rot_len:.3f}")
                    for i, (orig_angle, rot_angle) in enumerate(zip(original_angles, rotated_angles)):
                        print(f"  Angle {['α', 'β', 'γ'][i]}: Original = {orig_angle:.2f}, Rotated = {rot_angle:.2f}")

                    # Print rotation for individual lattice vectors
                    original_directions = [vec / np.linalg.norm(vec) for vec in struct2_std.lattice.matrix]
                    rotated_directions = [vec / np.linalg.norm(vec) for vec in rotated_lattice_vectors]
                    for i, (orig, rot) in enumerate(zip(original_directions, rotated_directions)):
                        dot_product = np.dot(orig, rot)
                        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
                        print(f"  Lattice vector {i+1} rotation: {angle:.2f} degrees")

                    # Create a copy of struct2 and update its lattice
                    modified_structure2 = copy.deepcopy(struct2_std)
                    modified_structure2.lattice = rotated_lattice

                    # Rotate Cartesian coordinates of sites
                    cartesian_coords = np.array([site.coords for site in modified_structure2.sites])
                    rotated_cartesian_coords = np.dot(cartesian_coords, rotation_matrix)
                    # Convert back to fractional coordinates in the new lattice
                    rotated_fractional_coords = rotated_lattice.get_fractional_coords(rotated_cartesian_coords)
                    wrapped_fractional_coords = rotated_fractional_coords % 1

                    # Update the modified structure with new coordinates
                    modified_structure2 = update_structure_with_rotated_coords(modified_structure2, wrapped_fractional_coords)

                    # Check for overlapping atoms after rotation
                    if has_overlapping_atoms(modified_structure2, tol=0.5):
                        print("Overlapping atoms detected after rotation. Returning None.")
                        return None

                    # Recalculate the orientation vector for the rotated structure
                    new_orientation_vector = get_normal_vector(modified_structure2, miller2)

                    # Check if the new orientation matches the orientation of struct1
                    if np.allclose(new_orientation_vector, normal_vector1, atol=0.1):
                        print("Orientation match achieved after rotation.")
                        aligned_structures.append({
                            "aligned_structure": modified_structure2,
                            "group1_index": group1["filtered_structure_index"],
                            "group2_index": group2["filtered_structure_index"]
                        })
                    else:
                        print("Orientation mismatch after rotation.")

    for i, aligned_structure_data in enumerate(aligned_structures):
        aligned_structure = aligned_structure_data["aligned_structure"]
        poscar_filename = f"aligned_structure_{i + 1}.vasp"
        aligned_structure.to(filename=poscar_filename, fmt="poscar")
        print(f"Aligned structure {i + 1} saved to {poscar_filename}")

    # If no aligned structure was found, return None.
    return aligned_structures if aligned_structures else None

#Step 6: If aligned, can they or their supercell have the same periodicity


if __name__ == '__main__':

    #crystal= Structure.from_file()
    #print(crystal)

    filtered_components, structure, atom_coordination= process_structure('../Main_fol_Zintl/./Binary/a1b1/14/Na1Sb1/CONTCAR')
    print(filtered_components)
    print('----')
    print(structure)
    print('----')
    print(atom_coordination)
