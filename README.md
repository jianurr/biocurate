# BioCurate

🔗 **Try it live:** [biocuratejian.streamlit.app](https://biocuratejian.streamlit.app/)

An open-source web tool for curating **any** bioactivity dataset — from ChEMBL,
PubChem, BindingDB, an in-house screen, or your own compiled spreadsheet — before
using it for QSAR / ML model training or downstream analysis.

It is target-agnostic and assay-agnostic: it doesn't assume EGFR, kinases, or any
specific project. You map your own column names to the pipeline's roles (SMILES,
activity value, units, assay type), and it works the same way on any dataset.

It flags — rather than silently drops or fixes — the most common quiet errors in
bioactivity data:

- Molecules that fail to parse or standardize
- Activity values with inconsistent or unrecognized units
- Exact duplicate compounds (by InChIKey)
- Near-duplicate compounds (Tanimoto similarity above a threshold)
- Compounds tested under more than one assay type (IC50 mixed with Ki, etc.)
- A simple per-row confidence score
- A chemical diversity / scaffold report

**Everything runs locally, in your browser, on your own machine — no data is
uploaded anywhere.** This matters if you're working with unpublished or proprietary
compound data.

## Contributing

This is meant to be a community tool. If you've hit a data-curation headache this
doesn't cover yet, see [CONTRIBUTING.md](CONTRIBUTING.md) — issues and pull requests
are welcome.


## 1. Setup (one-time)

You need Python 3.9+ installed. Then, from this folder:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the web app locally

```bash
streamlit run app.py
```

This opens a browser tab (usually at `http://localhost:8501`) with the upload interface.
Everything runs on your own laptop — no data leaves your machine.

## 3. Try it with the included sample data

Upload `sample_data.csv` (included in this folder) to see the pipeline in action on a
small, deliberately messy example dataset before trying it on your own ChEMBL export.

## 4. Using it from a Jupyter notebook instead

If you'd rather call the pipeline directly (e.g. inside a notebook):

```python
import pandas as pd
from pipeline import run_full_pipeline

df = pd.read_csv("your_data.csv")
result_df, report, near_dup_pairs = run_full_pipeline(
    df,
    smiles_col="smiles",
    value_col="value",
    unit_col="units",
    assay_type_col="assay_type",
)

print(report)
result_df.head()
```

## 5. Expected input format

A CSV with (at minimum) these four columns, named whatever you like — you map them
to the right roles in the app itself:

| SMILES column | activity value | units | assay type |
|---|---|---|---|
| `CCO` | `100` | `nM` | `IC50` |

## 6. Deploying it online (optional, later step)

Once you're happy with it locally, the simplest free option is
[Streamlit Community Cloud](https://streamlit.io/cloud) — push this folder to a public
GitHub repo and connect it there; no server management needed.

## 7. Project structure

```
biocurate/
├── app.py             # Streamlit web interface
├── pipeline.py         # Core curation logic (reusable, notebook-friendly)
├── requirements.txt
├── sample_data.csv      # small test dataset
└── README.md
```
