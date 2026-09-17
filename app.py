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

st.set_page_config(page_title="BioCurate", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
    }

    section.main > div {
        max-width: 1100px;
    }

    .bc-hero {
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(56, 189, 200, 0.18), rgba(80, 120, 220, 0.12));
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 1.6rem;
    }
    .bc-hero h1 {
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #5ee7df, #66a6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .bc-hero p {
        color: rgba(230,240,245,0.85);
        font-size: 1.02rem;
        max-width: 800px;
    }
    .bc-badges span {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        color: #d7ecff;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 12px 14px 6px 14px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    .bc-section-title {
        color: #cdeaff;
        font-weight: 600;
        font-size: 1.15rem;
        margin: 1.4rem 0 0.4rem 0;
        border-left: 4px solid #66a6ff;
        padding-left: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 BioCurate")
    st.markdown(
        "An open-source tool that flags — rather than silently drops or fixes — "
        "the quiet errors in bioactivity datasets before ML/QSAR work."
    )
    st.markdown("---")
    st.markdown("**What it checks:**")
    st.markdown(
        "- Structure parsing & standardization\n"
        "- Unit / activity conversion\n"
        "- Exact & near-duplicate compounds\n"
        "- Mixed assay-type compounds\n"
        "- Per-row confidence score\n"
        "- Scaffold diversity"
    )
    st.markdown("---")
    st.markdown("**Everything runs locally** — your data never leaves this session.")
    st.markdown("---")
    st.markdown("[⭐ View source / contribute on GitHub](https://github.com/jianurr/biocurate)")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="bc-hero">
        <h1>🧪 BioCurate</h1>
        <p>Upload a raw bioactivity dataset (ChEMBL, PubChem, BindingDB, or your own screen)
        and get a cleaned, standardized dataset plus a diagnostic report — with every
        structure, unit, duplicate, and assay-type issue flagged for you to inspect,
        never silently dropped.</p>
        <div class="bc-badges">
            <span>🧬 RDKit-powered</span>
            <span>🔓 Open source</span>
            <span>💻 Runs locally</span>
            <span>📊 Any bioactivity dataset</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])


def robust_read_csv(uploaded_file):
    """
    Try several common separators/encodings before giving up, and return
    a clear error message instead of letting pandas crash the app.
    """
    separators = [",", "\t", ";", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding, engine="python")
                if df.shape[1] > 1:  # a real delimiter was found, not one giant column
                    return df, None
            except Exception:
                continue

    # Last resort: let pandas skip broken rows instead of failing entirely
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
        return df, "Some malformed rows were skipped while reading the file."
    except Exception as e:
        return None, str(e)


if uploaded_file is not None:
    raw_df, read_warning = robust_read_csv(uploaded_file)

    if raw_df is None:
        st.error(
            "Could not read this file as a table. Please check that it is a plain "
            "CSV/TSV file with a single header row and consistent columns, then "
            "try again.\n\n"
            f"Technical detail: {read_warning}"
        )
        st.stop()

    if read_warning:
        st.warning(read_warning)

    st.markdown('<div class="bc-section-title">📋 Preview of uploaded data</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown('<div class="bc-section-title">🧭 Map your columns</div>', unsafe_allow_html=True)
    cols = raw_df.columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        smiles_col = st.selectbox("SMILES column", cols, index=0)
        value_col = st.selectbox("Activity value column", cols, index=min(1, len(cols) - 1))
    with c2:
        unit_col = st.selectbox("Units column", cols, index=min(2, len(cols) - 1))
        assay_type_col = st.selectbox("Assay type column (IC50/Ki/etc.)", cols, index=min(3, len(cols) - 1))

    st.markdown('<div class="bc-section-title">⚙️ Options</div>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)
    with o1:
        near_dup_threshold = st.slider("Near-duplicate similarity threshold (Tanimoto)", 0.80, 1.00, 0.95, 0.01)
    with o2:
        near_dup_cap = st.number_input(
            "Max compounds to check for near-duplicates (keeps this laptop-friendly)",
            min_value=50, max_value=2000, value=400, step=50
        )

    st.write("")
    run_button = st.button("🚀 Run curation pipeline", type="primary", use_container_width=True)

    if run_button:
        try:
            with st.spinner("Running pipeline..."):
                result_df, report, near_dup_pairs = run_full_pipeline(
                    raw_df, smiles_col, value_col, unit_col, assay_type_col,
                    near_dup_threshold=near_dup_threshold,
                    near_dup_cap=near_dup_cap,
                )
        except Exception as e:
            st.error(
                "Something went wrong while running the pipeline. This usually means "
                "the columns you mapped don't contain the kind of data expected "
                "(e.g. the 'SMILES column' should contain chemical structure strings, "
                "and the 'value' column should be numeric).\n\n"
                f"Technical detail: {e}"
            )
            st.stop()

        st.success("✅ Done — see the results below.")

        st.markdown('<div class="bc-section-title">📊 Summary report</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="bc-section-title">🚩 Flagged rows</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🧬 Structure/unit issues", "🔁 Duplicates", "⚖️ Mixed assay types"])

        with tab1:
            issues_df = result_df[
                (result_df["structure_status"] != "ok") | (result_df["unit_status"] != "ok")
            ]
            st.write(f"{len(issues_df)} rows flagged")
            st.dataframe(issues_df, use_container_width=True)

        with tab2:
            dup_df = result_df[result_df["exact_duplicate"]]
            st.write(f"{len(dup_df)} exact duplicate rows (by InChIKey)")
            st.dataframe(dup_df, use_container_width=True)

            if near_dup_pairs:
                st.write(f"{len(near_dup_pairs)} near-duplicate pairs found "
                         f"(checked first {report['near_duplicate_compounds_checked']} compounds)")
                near_dup_df = pd.DataFrame(near_dup_pairs, columns=["row_i", "row_j", "similarity"])
                st.dataframe(near_dup_df, use_container_width=True)

        with tab3:
            mixed_df = result_df[result_df["mixed_assay_flag"]]
            st.write(f"{len(mixed_df)} rows belong to compounds tested under more than one assay type")
            st.dataframe(mixed_df, use_container_width=True)

        st.divider()

        st.markdown('<div class="bc-section-title">🌈 Diversity overview</div>', unsafe_allow_html=True)
        scaffold_counts = result_df["scaffold"].value_counts().head(15)
        if not scaffold_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            scaffold_counts.plot(kind="bar", ax=ax, color="#66a6ff")
            ax.set_ylabel("Compound count", color="white")
            ax.set_title("Top 15 most common scaffolds", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            plt.xticks(rotation=75, ha="right", fontsize=7, color="white")
            plt.yticks(color="white")
            st.pyplot(fig, transparent=True)

        st.divider()

        st.markdown('<div class="bc-section-title">⬇️ Download results</div>', unsafe_allow_html=True)
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned dataset (CSV)",
            data=csv_bytes,
            file_name="curated_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("👆 Upload a CSV file to get started. Expected columns: SMILES, activity value, "
            "units (e.g. nM), and assay type (e.g. IC50/Ki).")

