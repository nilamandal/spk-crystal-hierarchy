import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ----------------------------
# Configuration
# ----------------------------
CSV_PATH = "AqSolDB_v1.0_min.csv"
OUT_PREFIX = "bench"
FRAC_TRAIN = 0.8
FRAC_VAL = 0.1
FRAC_TEST = 0.1


# ----------------------------
# Helpers
# ----------------------------
def generate_scaffold(smiles, include_chirality=True):
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles,
        includeChirality=include_chirality
    )


def molecule_is_valid(smiles):
    """Mimics filtering done implicitly during graph construction"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if mol.GetNumAtoms() == 0:
        return False
    if mol.GetNumBonds() == 0:
        return False
    return True


def scaffold_split(indices, smiles_list):
    """
    Deterministic scaffold split (DeepChem-style)
    """
    all_scaffolds = defaultdict(list)

    for idx in indices:
        scaffold = generate_scaffold(smiles_list[idx])
        all_scaffolds[scaffold].append(idx)

    # sort indices inside each scaffold
    all_scaffolds = {
        k: sorted(v) for k, v in all_scaffolds.items()
    }

    # sort scaffolds by size (desc), then by smallest index
    scaffold_sets = [
        s for _, s in sorted(
            all_scaffolds.items(),
            key=lambda x: (len(x[1]), x[1][0]),
            reverse=True
        )
    ]

    n_total = len(indices)
    train_cut = FRAC_TRAIN * n_total
    val_cut = (FRAC_TRAIN + FRAC_VAL) * n_total

    train_idx, val_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cut:
            train_idx.extend(scaffold_set)
        elif len(train_idx) + len(val_idx) + len(scaffold_set) <= val_cut:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    return train_idx, val_idx, test_idx


# ----------------------------
# Main
# ----------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    smiles = df["SMILES"].tolist()

    # Step 1: filter molecules FIRST
    valid_mask = [molecule_is_valid(s) for s in smiles]
    df_valid = df[valid_mask].reset_index(drop=True)
    smiles_valid = df_valid["SMILES"].tolist()

    print(f"Original rows: {len(df)}")
    print(f"Valid molecules kept: {len(df_valid)}")

    # Step 2: scaffold split on VALID molecules
    valid_indices = list(range(len(df_valid)))
    train_idx, val_idx, test_idx = scaffold_split(valid_indices, smiles_valid)

    # Sanity checks
    assert not set(train_idx) & set(val_idx)
    assert not set(train_idx) & set(test_idx)
    assert not set(val_idx) & set(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df_valid)

    # Step 3: reorder & save CSVs (retain all columns)
    df_train = df_valid.iloc[train_idx].reset_index(drop=True)
    df_val   = df_valid.iloc[val_idx].reset_index(drop=True)
    df_test  = df_valid.iloc[test_idx].reset_index(drop=True)

    df_train.to_csv(f"{OUT_PREFIX}_train.csv", index=False)
    df_val.to_csv(f"{OUT_PREFIX}_val.csv", index=False)
    df_test.to_csv(f"{OUT_PREFIX}_test.csv", index=False)

    print("Saved:")
    print(f"  {OUT_PREFIX}_train.csv ({len(df_train)})")
    print(f"  {OUT_PREFIX}_val.csv   ({len(df_val)})")
    print(f"  {OUT_PREFIX}_test.csv  ({len(df_test)})")


if __name__ == "__main__":
    main()

