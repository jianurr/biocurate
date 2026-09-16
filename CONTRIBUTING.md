# Contributing to BioCurate

Thanks for considering contributing! This project started as a small tool to clean up
bioactivity datasets before ML/QSAR work, and it's meant to grow with input from anyone
who's felt this pain.

## Ways to contribute

- **Report a bug** — open an issue with a minimal example (a few rows of data that
  trigger the problem, if possible).
- **Suggest a new check** — e.g. a new unit type, a PAINS/structural alert filter,
  a new assay-type normalization rule. Open an issue describing the idea first.
- **Submit a pull request** — for small fixes, feel free to go straight to a PR.
  For larger changes, please open an issue first to discuss the approach.

## Development setup

```bash
git clone https://github.com/<your-username>/biocurate.git
cd biocurate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Code structure

- `pipeline.py` — all curation logic. Keep this UI-independent so it stays usable
  from a notebook or a script, not just the web app.
- `app.py` — Streamlit UI only. Business logic should live in `pipeline.py`, not here.

## Style

- Keep functions small and focused (one check per function, as in the existing modules).
- Every transformation should be logged/flagged, never silently applied — this is
  the core design principle of the tool.
- Add a docstring to new functions explaining what they check and why it matters.

## Testing your changes

There's no formal test suite yet (contributions welcome here too!). At minimum, run
the pipeline against `sample_data.csv` and a real dataset you have access to, and
confirm the report numbers make sense before opening a PR.
