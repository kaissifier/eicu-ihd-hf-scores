import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_cohort.py", "02_population.py", "03_extract_first_24h.py", "04_scores.py", "05_performance.py",
         "06_calibration_belt.R", "07_cutoffs.py", "08_descriptives.py", "09_figures.py"]

for step in STEPS:
    cmd = [os.environ.get("RSCRIPT", "Rscript"), step] if step.endswith(".R") else [sys.executable, step]
    print(f"\n== {step}", flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
