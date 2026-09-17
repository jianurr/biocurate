"""
tests/test_pipeline.py

Run with:
    pytest tests/ -v

Covers the core pipeline functions with the same kinds of edge cases that
were manually stress-tested during development: missing values (None vs NaN),
invalid/empty SMILES, exact and near duplicates, and unit conversion edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from pipeline import (
    standardize_molecule,
    standardize_dataframe,
    convert_to_pActivity,
    normalize_units,
    flag_exact_duplicates,
    find_near_duplicates,
    flag_mixed_assay_types,
    compute_confidence_score,
    get_scaffold,
    guess_columns,
    run_qsar_cleaning,
)


# ---------------------------------------------------------------------------
# Module 1: Structure standardization
# ---------------------------------------------------------------------------

class TestStandardizeMolecule:
    def test_valid_smiles(self):
        smiles, inchikey, status, extra = standardize_molecule("CCO")
        assert status == "ok"
        assert smiles is not None
        assert inchikey is not None

    def test_invalid_smiles(self):
        smiles, inchikey, status, extra = standardize_molecule("not_a_real_smiles_@@@")
        assert status == "parse_failed"
        assert smiles is None

    def test_empty_string(self):
        smiles, inchikey, status, extra = standardize_molecule("")
        assert status == "empty_input"
        assert smiles is None

    def test_none_input(self):
        smiles, inchikey, status, extra = standardize_molecule(None)
        assert status == "empty_input"

    def test_same_molecule_different_smiles_gives_same_inchikey(self):
        # Two valid SMILES representations of the same molecule (ethanol)
        _, key1, _, _ = standardize_molecule("CCO")
        _, key2, _, _ = standardize_molecule("OCC")
        assert key1 == key2

    def test_salt_is_stripped(self):
        # Sodium acetate — salt remover should strip the Na+ counterion
        smiles, inchikey, status, extra = standardize_molecule("CC(=O)[O-].[Na+]")
        assert status == "ok"

    def test_tautomers_unify_when_enabled(self):
        # Keto vs enol form of acetone — should give the same InChIKey
        # ONLY when canonicalize_tautomers=True
        _, key_keto, _, _ = standardize_molecule("CC(=O)C", canonicalize_tautomers=True)
        _, key_enol, _, _ = standardize_molecule("CC(O)=C", canonicalize_tautomers=True)
        assert key_keto == key_enol

    def test_tautomers_not_unified_by_default(self):
        _, key_keto, _, _ = standardize_molecule("CC(=O)C")
        _, key_enol, _, _ = standardize_molecule("CC(O)=C")
        # Without canonicalization, these are (usually) different InChIKeys
        assert key_keto != key_enol

    def test_mixture_flagged(self):
        # Two unconnected real fragments neither of which is a known salt —
        # should be flagged as a mixture rather than silently kept
        _, _, status, extra = standardize_molecule("CCO.c1ccccc1")
        assert status == "ok"
        assert extra["is_mixture"] is True

    def test_single_component_not_flagged_as_mixture(self):
        _, _, status, extra = standardize_molecule("CCO")
        assert extra["is_mixture"] is False

    def test_ignore_stereo_option(self):
        # Two enantiomers should get different InChIKeys normally, but the
        # same no-stereo InChIKey when ignore_stereo=True
        _, key1, _, extra1 = standardize_molecule("C[C@H](N)C(=O)O", ignore_stereo=True)   # L-alanine
        _, key2, _, extra2 = standardize_molecule("C[C@@H](N)C(=O)O", ignore_stereo=True)  # D-alanine
        assert key1 != key2  # full InChIKey still distinguishes stereoisomers
        assert extra1["inchikey_no_stereo"] == extra2["inchikey_no_stereo"]  # but no-stereo version matches


# ---------------------------------------------------------------------------
# Module 2: Unit / activity normalization
# ---------------------------------------------------------------------------

class TestConvertToPActivity:
    def test_valid_nanomolar(self):
        pact, status = convert_to_pActivity(100, "nM")
        assert status == "ok"
        assert pact == pytest.approx(7.0, abs=0.01)  # -log10(100e-9) = 7

    def test_valid_micromolar(self):
        pact, status = convert_to_pActivity(1, "uM")
        assert status == "ok"
        assert pact == pytest.approx(6.0, abs=0.01)

    def test_none_value(self):
        pact, status = convert_to_pActivity(None, "nM")
        assert status == "missing_value"
        assert np.isnan(pact)

    def test_nan_value(self):
        # This is the specific bug caught during development: float(nan)
        # doesn't raise, so nan must be explicitly checked for.
        pact, status = convert_to_pActivity(float("nan"), "nM")
        assert status == "missing_value"
        assert np.isnan(pact)

    def test_negative_value(self):
        pact, status = convert_to_pActivity(-5, "nM")
        assert status == "non_positive_or_invalid_value"
        assert np.isnan(pact)

    def test_zero_value(self):
        pact, status = convert_to_pActivity(0, "nM")
        assert status == "non_positive_or_invalid_value"

    def test_unknown_unit(self):
        pact, status = convert_to_pActivity(100, "furlongs")
        assert "unknown_unit" in status
        assert np.isnan(pact)

    def test_non_numeric_value(self):
        pact, status = convert_to_pActivity("not_a_number", "nM")
        assert status == "non_numeric_value"

    def test_none_units(self):
        pact, status = convert_to_pActivity(100, None)
        assert "unknown_unit" in status


# ---------------------------------------------------------------------------
# Exact duplicate detection
# ---------------------------------------------------------------------------

class TestExactDuplicates:
    def test_detects_duplicates(self):
        df = pd.DataFrame({"inchikey": ["A", "A", "B", "C"]})
        result = flag_exact_duplicates(df)
        assert result["exact_duplicate"].tolist() == [True, True, False, False]

    def test_no_duplicates(self):
        df = pd.DataFrame({"inchikey": ["A", "B", "C"]})
        result = flag_exact_duplicates(df)
        assert not result["exact_duplicate"].any()


# ---------------------------------------------------------------------------
# Near-duplicate detection (both brute-force and scaffold-bucketed paths)
# ---------------------------------------------------------------------------

class TestNearDuplicates:
    def test_identical_molecules_are_near_duplicates(self):
        df = pd.DataFrame({"clean_smiles": ["CCO", "CCO", "c1ccccc1"]})
        pairs, n_comparisons, method = find_near_duplicates(df, threshold=0.95)
        assert method == "brute_force"
        assert len(pairs) >= 1  # the two CCO rows should match

    def test_dissimilar_molecules_not_flagged(self):
        df = pd.DataFrame({"clean_smiles": ["CCO", "c1ccc2ccccc2c1"]})  # ethanol vs naphthalene
        pairs, n_comparisons, method = find_near_duplicates(df, threshold=0.95)
        assert len(pairs) == 0

    def test_brute_force_used_below_limit(self):
        df = pd.DataFrame({"clean_smiles": ["CCO"] * 50})
        pairs, n_comparisons, method = find_near_duplicates(df, brute_force_limit=100)
        assert method == "brute_force"
        assert n_comparisons == 50 * 49 // 2  # exact, nothing skipped

    def test_scaffold_bucketed_used_above_limit(self):
        df = pd.DataFrame({"clean_smiles": ["CCO"] * 50})
        pairs, n_comparisons, method = find_near_duplicates(df, brute_force_limit=10)
        assert method == "scaffold_bucketed"

    def test_invalid_smiles_does_not_crash(self):
        df = pd.DataFrame({"clean_smiles": ["CCO", "invalid_xxx", None, "c1ccccc1"]})
        # Should not raise, even with unparseable/missing entries mixed in
        pairs, n_comparisons, method = find_near_duplicates(df, threshold=0.95)
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# Scaffold extraction
# ---------------------------------------------------------------------------

class TestScaffold:
    def test_valid_molecule_gets_scaffold(self):
        scaffold = get_scaffold("c1ccccc1CCN")  # phenethylamine-like
        assert scaffold is not None

    def test_invalid_smiles_returns_none(self):
        assert get_scaffold("not_valid") is None

    def test_none_input_returns_none(self):
        assert get_scaffold(None) is None


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------

class TestGuessColumns:
    def test_detects_named_columns(self):
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "c1ccccc1"],
            "standard_value": [100, 200],
            "standard_units": ["nM", "nM"],
            "standard_type": ["IC50", "Ki"],
        })
        guessed = guess_columns(df)
        assert guessed["smiles_col"] == "canonical_smiles"
        assert guessed["value_col"] == "standard_value"
        assert guessed["unit_col"] == "standard_units"
        assert guessed["assay_type_col"] == "standard_type"

    def test_detects_generic_columns_by_content(self):
        # No informative column names at all — must detect by content
        df = pd.DataFrame({
            "col_A": ["CCO", "c1ccccc1", "CCN(CC)CC"],
            "col_B": [100, 200, 80],
            "col_C": ["nM", "uM", "nM"],
            "col_D": ["IC50", "Ki", "IC50"],
        })
        guessed = guess_columns(df)
        assert guessed["smiles_col"] == "col_A"
        assert guessed["unit_col"] == "col_C"
        assert guessed["assay_type_col"] == "col_D"


# ---------------------------------------------------------------------------
# End-to-end pipeline (the function the app actually calls)
# ---------------------------------------------------------------------------

class TestRunQsarCleaning:
    @pytest.fixture
    def messy_df(self):
        return pd.DataFrame({
            "id": ["A", "B", "C", "D", "E", "F"],
            "smiles": ["CCO", "CCO", "c1ccccc1", None, "invalid_xxx", "CCN(CC)CC"],
            "value": [100, 100, 50, 200, 300, None],
            "units": ["nM", "nM", "uM", "nM", "nM", "nM"],
            "assay_type": ["IC50", "IC50", "Ki", "IC50", "IC50", "Ki"],
            "irrelevant_column": ["x"] * 6,
        })

    def test_strips_irrelevant_columns(self, messy_df):
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            messy_df, "smiles", "value", "units", "assay_type", id_col="id",
        )
        assert "irrelevant_column" not in clean_df.columns
        assert report["columns_removed"] == 1

    def test_removes_missing_and_invalid_rows(self, messy_df):
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            messy_df, "smiles", "value", "units", "assay_type", id_col="id",
        )
        # D (missing smiles), E (invalid smiles), F (missing value) should all be removed
        assert set(removed_df["id"]) == {"D", "E", "F"}

    def test_removes_exact_duplicates(self, messy_df):
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            messy_df, "smiles", "value", "units", "assay_type", id_col="id",
        )
        # B is an exact duplicate of A
        assert "B" in duplicates_df["id"].values

    def test_no_nan_in_final_output(self, messy_df):
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            messy_df, "smiles", "value", "units", "assay_type", id_col="id",
        )
        assert not clean_df["pActivity"].isna().any()

    def test_no_duplicate_structures_in_final_output(self, messy_df):
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            messy_df, "smiles", "value", "units", "assay_type", id_col="id",
        )
        assert not clean_df["inchikey"].duplicated().any()

    def test_report_numbers_are_internally_consistent(self, messy_df):
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            messy_df, "smiles", "value", "units", "assay_type", id_col="id",
        )
        assert report["original_rows"] == len(messy_df)
        assert len(clean_df) == report["final_clean_rows"]
        assert len(removed_df) == report["rows_removed_missing_or_invalid"]

    def test_works_without_optional_columns(self):
        # id_col and assay_type_col are both optional — should still run
        df = pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1"],
            "value": [100, 200],
            "units": ["nM", "nM"],
        })
        clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
            df, "smiles", "value", "units", assay_type_col=None, id_col=None,
        )
        assert len(clean_df) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
