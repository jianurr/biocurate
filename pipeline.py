"""
pipeline.py
Core bioactivity data curation logic, kept separate from the UI (app.py)
so it can also be imported and used directly in a Jupyter notebook.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

UNIT_TO_MOLAR = {
    "nM": 1e-9,
    "uM": 1e-6,
    "µM": 1e-6,
    "mM": 1e-3,
    "M": 1.0,
}

_remover = SaltRemover()


# ---------- Module 1: Structure standardization ----------

def standardize_molecule(smiles: str):
    """Returns (clean_smiles, inchikey, status)."""
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None, None, "empty_input"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "parse_failed"

    try:
        mol = _remover.StripMol(mol, dontRemoveEverything=True)
        clean_smiles = Chem.MolToSmiles(mol)
        inchikey = Chem.MolToInchiKey(mol)
        return clean_smiles, inchikey, "ok"
    except Exception as e:
        return None, None, f"standardization_failed: {e}"


def standardize_dataframe(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    df = df.copy()
    results = df[smiles_col].apply(standardize_molecule)
    df["clean_smiles"] = results.apply(lambda x: x[0])
    df["inchikey"] = results.apply(lambda x: x[1])
    df["structure_status"] = results.apply(lambda x: x[2])
    return df


# ---------- Module 2: Unit / activity normalization ----------

def convert_to_pActivity(value, units):
    """Convert a concentration value to -log10(M). Returns (pActivity, status)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan, "non_numeric_value"

    if units not in UNIT_TO_MOLAR:
        return np.nan, f"unknown_unit:{units}"

    molar_value = value * UNIT_TO_MOLAR[units]
    if molar_value <= 0:
        return np.nan, "non_positive_value"

    return -np.log10(molar_value), "ok"


def normalize_units(df: pd.DataFrame, value_col: str, unit_col: str) -> pd.DataFrame:
    df = df.copy()
    results = df.apply(lambda row: convert_to_pActivity(row[value_col], row[unit_col]), axis=1)
    df["pActivity"] = results.apply(lambda x: x[0])
    df["unit_status"] = results.apply(lambda x: x[1])
    return df


# ---------- Module 3: Duplicate detection ----------

def flag_exact_duplicates(df: pd.DataFrame, key_col: str = "inchikey") -> pd.DataFrame:
    df = df.copy()
    counts = df[key_col].value_counts()
    dup_keys = counts[counts > 1].index
    df["exact_duplicate"] = df[key_col].isin(dup_keys)
    return df


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def find_near_duplicates(df: pd.DataFrame, smiles_col: str = "clean_smiles",
                          threshold: float = 0.95, max_compounds: int = 400):
    """
    O(n^2) pairwise comparison — capped at max_compounds to stay laptop-friendly.
    Returns a list of (index_i, index_j, similarity) tuples.
    """
    subset = df.head(max_compounds).reset_index(drop=True)
    fps = subset[smiles_col].apply(get_fingerprint).tolist()

    pairs = []
    for i in range(len(fps)):
        if fps[i] is None:
            continue
        for j in range(i + 1, len(fps)):
            if fps[j] is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim >= threshold:
                pairs.append((i, j, round(sim, 3)))
    return pairs, len(subset)


# ---------- Module 4: Assay type consistency ----------

def flag_mixed_assay_types(df: pd.DataFrame, key_col: str = "inchikey",
                            assay_type_col: str = "standard_type") -> pd.DataFrame:
    df = df.copy()
    n_types = df.groupby(key_col)[assay_type_col].nunique()
    mixed_keys = n_types[n_types > 1].index
    df["mixed_assay_flag"] = df[key_col].isin(mixed_keys)
    return df


# ---------- Module 5: Simple confidence score ----------

def compute_confidence_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score = pd.Series(1.0, index=df.index)

    if "structure_status" in df.columns:
        score -= (df["structure_status"] != "ok") * 0.5
    if "unit_status" in df.columns:
        score -= (df["unit_status"] != "ok") * 0.3
    if "exact_duplicate" in df.columns:
        score -= df["exact_duplicate"].astype(bool) * 0.1
    if "mixed_assay_flag" in df.columns:
        score -= df["mixed_assay_flag"].astype(bool) * 0.2

    df["confidence_score"] = score.clip(lower=0.0, upper=1.0)
    return df


# ---------- Module 6: Diversity / scaffold report ----------

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def add_scaffolds(df: pd.DataFrame, smiles_col: str = "clean_smiles") -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(get_scaffold)
    return df


# ---------- Orchestration ----------

def run_full_pipeline(df: pd.DataFrame, smiles_col: str, value_col: str,
                       unit_col: str, assay_type_col: str,
                       near_dup_threshold: float = 0.95,
                       near_dup_cap: int = 400):
    """
    Runs all modules in sequence and returns (result_df, report_dict, near_dup_pairs).
    """
    df = standardize_dataframe(df, smiles_col)
    df = normalize_units(df, value_col, unit_col)
    df = flag_exact_duplicates(df)
    df = flag_mixed_assay_types(df, assay_type_col=assay_type_col)
    df = compute_confidence_score(df)
    df = add_scaffolds(df)

    near_dup_pairs, n_checked = find_near_duplicates(
        df.dropna(subset=["clean_smiles"]), threshold=near_dup_threshold,
        max_compounds=near_dup_cap
    )

    n_total = len(df)
    n_struct_fail = (df["structure_status"] != "ok").sum()
    n_unit_fail = (df["unit_status"] != "ok").sum()
    n_exact_dup = df["exact_duplicate"].sum()
    n_mixed_assay = df["mixed_assay_flag"].sum()
    n_scaffolds = df["scaffold"].nunique(dropna=True)
    n_valid_structures = df["clean_smiles"].notna().sum()

    report = {
        "total_rows": n_total,
        "structure_parse_failures": int(n_struct_fail),
        "unit_conversion_failures": int(n_unit_fail),
        "exact_duplicates": int(n_exact_dup),
        "near_duplicate_pairs": len(near_dup_pairs),
        "near_duplicate_compounds_checked": n_checked,
        "mixed_assay_compounds": int(n_mixed_assay),
        "unique_scaffolds": int(n_scaffolds),
        "diversity_ratio": round(n_scaffolds / n_valid_structures, 3) if n_valid_structures else 0.0,
        "mean_confidence_score": round(df["confidence_score"].mean(), 3),
    }

    return df, report, near_dup_pairs
