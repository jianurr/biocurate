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
    if pd.isna(value):
        return np.nan, "missing_value"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan, "non_numeric_value"

    if pd.isna(value):
        return np.nan, "missing_value"

    if pd.isna(units) or units not in UNIT_TO_MOLAR:
        return np.nan, f"unknown_unit:{units}"

    molar_value = value * UNIT_TO_MOLAR[units]
    if not np.isfinite(molar_value) or molar_value <= 0:
        return np.nan, "non_positive_or_invalid_value"

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


# ---------- Automatic column detection ----------

def guess_columns(df: pd.DataFrame):
    """
    Best-effort automatic detection of which column is SMILES, activity value,
    units, and assay type — based on column names first, then content as a
    fallback. Returns a dict of column names (or None if nothing confident
    was found for that role).
    """
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}

    def find_by_keywords(keywords, exclude=None):
        exclude = exclude or []
        for c in cols:
            name = lower_map[c]
            if c in exclude:
                continue
            if any(k in name for k in keywords):
                return c
        return None

    guessed = {}

    # SMILES: name-based first, then content-based (try parsing a sample)
    smiles_col = find_by_keywords(["smiles", "canonical_smiles", "structure"])
    if smiles_col is None:
        best_col, best_hits = None, 0
        for c in cols:
            sample = df[c].dropna().astype(str).head(20)
            hits = sum(1 for s in sample if Chem.MolFromSmiles(s) is not None)
            if hits > best_hits and hits >= max(3, len(sample) // 2):
                best_col, best_hits = c, hits
        smiles_col = best_col
    guessed["smiles_col"] = smiles_col

    # Activity value: name-based first, then "numeric column that isn't smiles"
    value_col = find_by_keywords(
        ["standard_value", "activity_value", "ic50", "ki", "ec50", "kd", "value", "activity", "affinity"],
        exclude=[smiles_col] if smiles_col else [],
    )
    if value_col is None:
        numeric_cols = [c for c in cols if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])]
        value_col = numeric_cols[0] if numeric_cols else None
    guessed["value_col"] = value_col

    # Units
    unit_col = find_by_keywords(["unit", "units"], exclude=[smiles_col, value_col])
    if unit_col is None:
        known_units = {"nm", "um", "µm", "mm", "m"}
        for c in cols:
            if c in (smiles_col, value_col):
                continue
            sample = df[c].dropna().astype(str).str.lower().head(20)
            if sample.isin(known_units).sum() >= max(1, len(sample) // 2):
                unit_col = c
                break
    guessed["unit_col"] = unit_col

    # Assay type (optional)
    assay_col = find_by_keywords(
        ["assay_type", "standard_type", "assay", "type"],
        exclude=[smiles_col, value_col, unit_col],
    )
    if assay_col is None:
        known_types = {"ic50", "ki", "ec50", "kd"}
        for c in cols:
            if c in (smiles_col, value_col, unit_col):
                continue
            sample = df[c].dropna().astype(str).str.lower().head(20)
            if sample.isin(known_types).sum() >= max(1, len(sample) // 2):
                assay_col = c
                break
    guessed["assay_type_col"] = assay_col

    return guessed


# ---------- QSAR-ready cleaning orchestration ----------

def run_qsar_cleaning(df: pd.DataFrame, smiles_col: str, value_col: str,
                       unit_col: str, assay_type_col: str = None,
                       id_col: str = None, remove_near_duplicates: bool = False,
                       near_dup_threshold: float = 0.98, near_dup_cap: int = 500):
    """
    End-to-end automatic cleaning for ML/QSAR-ready output.

    Given a raw dataset with any number of extra/unnecessary columns, this:
      1. Keeps only the essential columns (id, smiles, value, units, assay type)
      2. Standardizes structures (drops rows with missing/invalid SMILES)
      3. Converts activity values to a consistent pActivity scale
         (drops rows with missing/invalid activity values or units)
      4. Removes exact duplicate compounds (keeps the first occurrence)
      5. Optionally removes near-duplicate compounds
      6. Optionally flags mixed assay-type compounds (kept, just flagged)

    Returns:
        clean_df       — the final ML-ready dataset
        removed_df     — rows dropped for missing/invalid SMILES or activity data,
                          with a 'removal_reason' column
        duplicates_df  — rows dropped as exact (or near) duplicates,
                          with a 'removal_reason' column
        report         — summary dict of what happened at each step
    """
    df = df.copy()
    original_n = len(df)

    # ---- Step 1: keep only essential columns ----
    keep_cols = [c for c in [id_col, smiles_col, value_col, unit_col, assay_type_col] if c]
    keep_cols = list(dict.fromkeys(keep_cols))  # dedupe, preserve order
    dropped_columns = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols].copy()

    # ---- Step 2: standardize structures ----
    df = standardize_dataframe(df, smiles_col)

    # ---- Step 3: normalize units / activity values ----
    df = normalize_units(df, value_col, unit_col)

    # ---- Step 4: split off rows with missing/invalid critical data ----
    invalid_structure_mask = df["structure_status"] != "ok"
    invalid_value_mask = (df["unit_status"] != "ok") | df["pActivity"].isna()
    invalid_mask = invalid_structure_mask | invalid_value_mask

    removed_df = df[invalid_mask].copy()

    def _removal_reason(row):
        reasons = []
        if row["structure_status"] != "ok":
            reasons.append(f"invalid_structure:{row['structure_status']}")
        if row["unit_status"] != "ok":
            reasons.append(f"invalid_activity_value:{row['unit_status']}")
        return "; ".join(reasons)

    if not removed_df.empty:
        removed_df["removal_reason"] = removed_df.apply(_removal_reason, axis=1)

    df = df[~invalid_mask].copy()
    n_after_validity_filter = len(df)

    # ---- Step 5: exact duplicate removal (keep first occurrence) ----
    df = flag_exact_duplicates(df, key_col="inchikey")
    exact_dup_mask = df["exact_duplicate"] & df.duplicated(subset=["inchikey"], keep="first")
    duplicates_df = df[exact_dup_mask].copy()
    if not duplicates_df.empty:
        duplicates_df["removal_reason"] = "exact_duplicate_structure"

    df = df[~exact_dup_mask].copy()
    n_after_exact_dedup = len(df)

    # ---- Step 6: optional near-duplicate removal ----
    near_dup_removed_count = 0
    if remove_near_duplicates and len(df) > 1:
        near_dup_pairs, n_checked = find_near_duplicates(
            df, smiles_col="clean_smiles",
            threshold=near_dup_threshold, max_compounds=near_dup_cap
        )
        # Drop the second compound in each near-duplicate pair (keep the first)
        subset = df.head(near_dup_cap).reset_index()  # 'index' = original df index
        drop_positions = {j for (_, j, _) in near_dup_pairs}
        drop_original_indices = subset.loc[list(drop_positions), "index"].tolist() if drop_positions else []

        if drop_original_indices:
            near_dup_rows = df.loc[drop_original_indices].copy()
            near_dup_rows["removal_reason"] = "near_duplicate_structure"
            duplicates_df = pd.concat([duplicates_df, near_dup_rows], ignore_index=True)
            df = df.drop(index=drop_original_indices)
            near_dup_removed_count = len(drop_original_indices)

    # ---- Step 7: optional mixed assay-type flag (kept, not removed) ----
    if assay_type_col:
        df = flag_mixed_assay_types(df, key_col="inchikey", assay_type_col=assay_type_col)

    # ---- Final clean output: tidy column selection ----
    final_cols = []
    if id_col:
        final_cols.append(id_col)
    final_cols += ["clean_smiles", "inchikey"]
    if assay_type_col:
        final_cols.append(assay_type_col)
    final_cols += ["pActivity"]
    if assay_type_col and "mixed_assay_flag" in df.columns:
        final_cols.append("mixed_assay_flag")

    clean_df = df[[c for c in final_cols if c in df.columns]].reset_index(drop=True)
    clean_df = clean_df.rename(columns={"clean_smiles": "smiles"})

    report = {
        "original_rows": original_n,
        "columns_removed": len(dropped_columns),
        "columns_removed_names": dropped_columns,
        "rows_removed_missing_or_invalid": int(len(removed_df)),
        "rows_after_validity_filter": n_after_validity_filter,
        "exact_duplicates_removed": int(len(duplicates_df)) - near_dup_removed_count if not duplicates_df.empty else 0,
        "near_duplicates_removed": near_dup_removed_count,
        "final_clean_rows": len(clean_df),
        "percent_retained": round(100 * len(clean_df) / original_n, 1) if original_n else 0.0,
    }

    return clean_df, removed_df, duplicates_df, report


# ---------- Orchestration (diagnostic / exploratory report — original mode) ----------

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
