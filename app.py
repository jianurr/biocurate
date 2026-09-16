"""
app.py
Streamlit web interface for the bioactivity data curation pipeline.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pipeline import run_full_pipeline

st.set_page_config(page_title="BioCurate", layout="wide")

st.title("🧪 BioCurate — Bioactivity Data Curation Tool")
st.markdown(
    "Upload a raw bioactivity dataset (e.g. exported from ChEMBL/PubChem/BindingDB) "
    "and get a cleaned, standardized dataset plus a diagnostic report — "
    "structure issues, unit mismatches, duplicates, mixed assay types, and chemical diversity, "
    "all flagged rather than silently dropped."
)

st.divider()

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(raw_df.head(10))

    st.subheader("Map your columns")
    cols = raw_df.columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        smiles_col = st.selectbox("SMILES column", cols, index=0)
        value_col = st.selectbox("Activity value column", cols, index=min(1, len(cols) - 1))
    with c2:
        unit_col = st.selectbox("Units column", cols, index=min(2, len(cols) - 1))
        assay_type_col = st.selectbox("Assay type column (IC50/Ki/etc.)", cols, index=min(3, len(cols) - 1))

    st.subheader("Options")
    near_dup_threshold = st.slider("Near-duplicate similarity threshold (Tanimoto)", 0.80, 1.00, 0.95, 0.01)
    near_dup_cap = st.number_input(
        "Max compounds to check for near-duplicates (keeps this laptop-friendly)",
        min_value=50, max_value=2000, value=400, step=50
    )

    run_button = st.button("Run curation pipeline", type="primary")

    if run_button:
        with st.spinner("Running pipeline..."):
            result_df, report, near_dup_pairs = run_full_pipeline(
                raw_df, smiles_col, value_col, unit_col, assay_type_col,
                near_dup_threshold=near_dup_threshold,
                near_dup_cap=near_dup_cap,
            )

        st.success("Done.")

        st.subheader("Summary report")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Total rows", report["total_rows"])
        rc1.metric("Structure parse failures", report["structure_parse_failures"])
        rc2.metric("Unit conversion failures", report["unit_conversion_failures"])
        rc2.metric("Exact duplicates", report["exact_duplicates"])
        rc3.metric("Near-duplicate pairs", report["near_duplicate_pairs"])
        rc3.metric("Mixed assay-type compounds", report["mixed_assay_compounds"])
        rc4.metric("Unique scaffolds", report["unique_scaffolds"])
        rc4.metric("Diversity ratio", report["diversity_ratio"])

        st.metric("Mean confidence score", report["mean_confidence_score"])

        st.divider()

        st.subheader("Flagged rows")
        tab1, tab2, tab3 = st.tabs(["Structure/unit issues", "Duplicates", "Mixed assay types"])

        with tab1:
            issues_df = result_df[
                (result_df["structure_status"] != "ok") | (result_df["unit_status"] != "ok")
            ]
            st.write(f"{len(issues_df)} rows flagged")
            st.dataframe(issues_df)

        with tab2:
            dup_df = result_df[result_df["exact_duplicate"]]
            st.write(f"{len(dup_df)} exact duplicate rows (by InChIKey)")
            st.dataframe(dup_df)

            if near_dup_pairs:
                st.write(f"{len(near_dup_pairs)} near-duplicate pairs found "
                         f"(checked first {report['near_duplicate_compounds_checked']} compounds)")
                near_dup_df = pd.DataFrame(near_dup_pairs, columns=["row_i", "row_j", "similarity"])
                st.dataframe(near_dup_df)

        with tab3:
            mixed_df = result_df[result_df["mixed_assay_flag"]]
            st.write(f"{len(mixed_df)} rows belong to compounds tested under more than one assay type")
            st.dataframe(mixed_df)

        st.divider()

        st.subheader("Diversity overview")
        scaffold_counts = result_df["scaffold"].value_counts().head(15)
        if not scaffold_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            scaffold_counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Compound count")
            ax.set_title("Top 15 most common scaffolds")
            plt.xticks(rotation=75, ha="right", fontsize=7)
            st.pyplot(fig)

        st.divider()

        st.subheader("Download results")
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned dataset (CSV)",
            data=csv_bytes,
            file_name="curated_dataset.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a CSV file to get started. Expected columns: SMILES, activity value, "
            "units (e.g. nM), and assay type (e.g. IC50/Ki).")
