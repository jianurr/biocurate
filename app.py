"""
app.py
Streamlit web interface for BioCurate — automatic bioactivity data cleaning
for ML/QSAR-ready datasets.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from rdkit import Chem

from pipeline import guess_columns, run_qsar_cleaning

st.set_page_config(page_title="BioCurate", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling — light, high-contrast theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #f4f9fc 0%, #eaf3fb 50%, #eef6f2 100%);
    }
    section.main > div { max-width: 1100px; }

    .bc-hero {
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #ffffff, #eef7fb);
        border: 1px solid rgba(30, 60, 90, 0.10);
        box-shadow: 0 4px 18px rgba(20, 40, 70, 0.06);
        margin-bottom: 1.6rem;
    }
    .bc-hero h1 { font-size: 2.3rem; margin-bottom: 0.3rem; color: #14395c; }
    .bc-hero p { color: #33475b; font-size: 1.02rem; max-width: 820px; }
    .bc-badges span {
        display: inline-block; background: #eaf6ff; border: 1px solid #bfe0f5;
        color: #1c5f8a; padding: 3px 11px; border-radius: 999px;
        font-size: 0.78rem; margin-right: 6px; margin-top: 8px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid rgba(30, 60, 90, 0.10);
        border-radius: 12px; padding: 12px 14px 6px 14px;
        box-shadow: 0 2px 8px rgba(20, 40, 70, 0.04);
    }
    .bc-section-title {
        color: #14395c; font-weight: 600; font-size: 1.15rem;
        margin: 1.4rem 0 0.4rem 0; border-left: 4px solid #3d8bc4; padding-left: 10px;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
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
        "Upload any raw bioactivity export (ChEMBL, PubChem, BindingDB, or your own "
        "screen) and get back a clean, ML/QSAR-ready dataset automatically."
    )
    st.markdown("---")
    st.markdown("**What happens automatically:**")
    st.markdown(
        "1. Detects the SMILES, activity value, units, and assay-type columns\n"
        "2. Drops every other (unneeded) column\n"
        "3. Standardizes chemical structures\n"
        "4. Converts activity values to a consistent scale (pActivity)\n"
        "5. Removes rows with missing/invalid structures or values\n"
        "6. Removes exact duplicate compounds\n"
        "7. Optionally removes near-duplicate compounds"
    )
    st.markdown("---")
    st.markdown("**Nothing is silently discarded** — removed and duplicate rows "
                "are given back to you as separate downloadable files.")
    st.markdown("---")
    st.markdown("**Runs locally** — your data isn't stored anywhere.")
    st.markdown("---")
    st.markdown("[⭐ Source / contribute on GitHub](https://github.com/jianurr/biocurate)")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="bc-hero">
        <h1>🧪 BioCurate</h1>
        <p>Upload a raw bioactivity dataset. BioCurate automatically detects the
        columns that matter, strips out everything else, fixes and standardizes
        the chemistry, removes duplicates, and hands you back a clean file that's
        ready for QSAR/ML training — plus the rows it removed, kept separately
        so nothing disappears without a trace.</p>
        <div class="bc-badges">
            <span>🧬 RDKit-powered</span>
            <span>🔓 Open source</span>
            <span>💻 Runs locally</span>
            <span>⚡ Fully automatic</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])


def robust_read_csv(uploaded_file):
    """Try common separators/encodings; return (df, warning) or (None, error)."""
    separators = [",", "\t", ";", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding, engine="python")
                if df.shape[1] > 1:
                    return df, None
            except Exception:
                continue

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
            "CSV/TSV file with a single header row and consistent columns.\n\n"
            f"Technical detail: {read_warning}"
        )
        st.stop()

    if read_warning:
        st.warning(read_warning)

    st.markdown('<div class="bc-section-title">📋 Preview of uploaded data</div>', unsafe_allow_html=True)
    st.caption(f"{raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    st.dataframe(raw_df.head(10), use_container_width=True)

    # ---- Automatic column detection ----
    guessed = guess_columns(raw_df)
    cols = raw_df.columns.tolist()

    missing_roles = [role for role, col in guessed.items() if col is None and role != "id_col"]

    st.markdown('<div class="bc-section-title">🧭 Column detection</div>', unsafe_allow_html=True)
    if not missing_roles:
        st.success(
            f"Detected automatically — SMILES: **{guessed['smiles_col']}**, "
            f"Value: **{guessed['value_col']}**, "
            f"Units: **{guessed['unit_col']}**, "
            f"Assay type: **{guessed['assay_type_col'] or 'not found (optional)'}**"
        )
    else:
        st.warning(
            "Couldn't confidently detect every required column automatically "
            f"({', '.join(missing_roles)}). Please check/fix the mapping below."
        )

    with st.expander("🔧 Adjust column mapping (only needed if detection looks wrong)"):
        c1, c2 = st.columns(2)
        with c1:
            smiles_col = st.selectbox(
                "SMILES column", cols,
                index=cols.index(guessed["smiles_col"]) if guessed["smiles_col"] in cols else 0,
            )
            value_col = st.selectbox(
                "Activity value column", cols,
                index=cols.index(guessed["value_col"]) if guessed["value_col"] in cols else min(1, len(cols) - 1),
            )
        with c2:
            unit_col = st.selectbox(
                "Units column", cols,
                index=cols.index(guessed["unit_col"]) if guessed["unit_col"] in cols else min(2, len(cols) - 1),
            )
            assay_options = ["(none)"] + cols
            default_assay = guessed["assay_type_col"] if guessed["assay_type_col"] in cols else "(none)"
            assay_type_col = st.selectbox(
                "Assay type column (optional)", assay_options,
                index=assay_options.index(default_assay),
            )
            assay_type_col = None if assay_type_col == "(none)" else assay_type_col

        id_options = ["(none)"] + cols
        id_col = st.selectbox("Compound ID column (optional)", id_options, index=0)
        id_col = None if id_col == "(none)" else id_col

    st.markdown('<div class="bc-section-title">⚙️ Options</div>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)
    with o1:
        remove_near_dup = st.checkbox(
            "Remove near-duplicate compounds (Tanimoto ≥ 0.98)",
            value=False,
            help="Off by default — near-duplicates can sometimes be legitimate close "
                 "analogs, so this is opt-in.",
        )
        canonicalize_tautomers = st.checkbox(
            "Unify tautomers (e.g. keto/enol forms) as the same compound",
            value=False,
            help="Off by default. When on, tautomers are canonicalized before "
                 "deduplication — a real chemistry choice about what counts as "
                 "'the same compound', not just a bug fix.",
        )
    with o2:
        ignore_stereo = st.checkbox(
            "Treat stereoisomers as duplicates of each other",
            value=False,
            help="Off by default — stereoisomers (e.g. enantiomers) are treated as "
                 "distinct compounds unless you explicitly opt into merging them.",
        )

    st.write("")
    run_button = st.button("🚀 Clean my data", type="primary", use_container_width=True)

    if run_button:
        try:
            with st.spinner("Cleaning your data..."):
                clean_df, removed_df, duplicates_df, report = run_qsar_cleaning(
                    raw_df, smiles_col, value_col, unit_col, assay_type_col,
                    id_col=id_col, remove_near_duplicates=remove_near_dup,
                    canonicalize_tautomers=canonicalize_tautomers,
                    ignore_stereo=ignore_stereo,
                )
        except Exception as e:
            st.error(
                "Something went wrong while cleaning. This usually means the columns "
                "mapped above don't contain the kind of data expected (e.g. the SMILES "
                "column should contain chemical structure strings, the value column "
                "should be numeric).\n\n"
                f"Technical detail: {e}"
            )
            st.stop()

        st.success("✅ Done — see the results below.")

        st.markdown('<div class="bc-section-title">📊 Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original rows", report["original_rows"])
        m1.metric("Columns removed", report["columns_removed"])
        m2.metric("Removed (missing/invalid)", report["rows_removed_missing_or_invalid"])
        m2.metric("Exact duplicates removed", report["exact_duplicates_removed"])
        m3.metric("Near-duplicates removed", report["near_duplicates_removed"])
        if report.get("near_duplicate_method"):
            method_label = "exact (brute-force)" if report["near_duplicate_method"] == "brute_force" else "fast (scaffold-bucketed)"
            st.caption(f"Near-duplicate check used the **{method_label}** method "
                       f"({'guaranteed complete' if report['near_duplicate_method'] == 'brute_force' else 'may miss a small number of cross-scaffold near-duplicates in exchange for speed on large datasets'}).")
        m3.metric("Final clean rows", report["final_clean_rows"])
        m4.metric("Data retained", f"{report['percent_retained']}%")
        m4.metric("Mixtures flagged", report["mixtures_flagged"])

        if report["columns_removed_names"]:
            with st.expander(f"📎 {report['columns_removed']} columns removed (click to see which)"):
                st.write(", ".join(report["columns_removed_names"]))

        st.divider()

        st.markdown('<div class="bc-section-title">📥 Results</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs([
            f"✅ Clean dataset ({len(clean_df)})",
            f"🚫 Removed — missing/invalid ({len(removed_df)})",
            f"🔁 Removed — duplicates ({len(duplicates_df)})",
        ])

        with tab1:
            st.caption("Ready for QSAR/ML training — standardized SMILES, consistent pActivity scale.")
            st.dataframe(clean_df, use_container_width=True)
            st.download_button(
                "⬇️ Download cleaned dataset (CSV)",
                data=clean_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with tab2:
            st.caption("Rows dropped because of missing or unparseable SMILES, or missing/invalid activity values.")
            if removed_df.empty:
                st.info("No rows were removed for missing/invalid data.")
            else:
                st.dataframe(removed_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download removed (missing/invalid) rows (CSV)",
                    data=removed_df.to_csv(index=False).encode("utf-8"),
                    file_name="removed_missing_or_invalid.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with tab3:
            st.caption("Duplicate compounds (exact, and optionally near-duplicate) — first occurrence was kept in the clean dataset.")
            if duplicates_df.empty:
                st.info("No duplicate compounds were found.")
            else:
                st.dataframe(duplicates_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download removed duplicate rows (CSV)",
                    data=duplicates_df.to_csv(index=False).encode("utf-8"),
                    file_name="removed_duplicates.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

else:
    st.info("👆 Upload a CSV file to get started. BioCurate will auto-detect the "
            "SMILES, activity value, units, and assay-type columns for you.")
