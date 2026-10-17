import os
from pathlib import Path

EICU_DIR = Path(os.environ.get("EICU_DIR", "data/eicu-crd"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "output"))
EXTRACT_DIR = OUT_DIR / "extract"
FIG_DIR = OUT_DIR / "figures"
SEED = 20261016

# predicted probability column of each score
SCORES = {"APACHE IVa": "p_apache", "SAPS II": "sapsii_prob", "OASIS": "oasis_prob"}


def table(name):
    """Path of an eICU-CRD table, gzipped or not, directly under EICU_DIR or in a subfolder."""
    for path in (EICU_DIR / f"{name}.csv.gz", EICU_DIR / f"{name}.csv"):
        if path.is_file():
            return path
    found = sorted((p for p in EICU_DIR.rglob(f"{name}.csv*") if p.is_file()), key=lambda p: (len(p.parts), str(p)))
    if not found:
        raise FileNotFoundError(f"{name}.csv not found under {EICU_DIR}")
    return found[0]


def out(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / name
