# eicu-ihd-hf-scores

Code for the study comparing APACHE IVa, SAPS II and OASIS in patients with ischemic heart disease admitted to the
ICU with acute or decompensated heart failure, using the eICU Collaborative Research Database (eICU-CRD).

Authors: Tiago da Conceição Aquino, Lucas Alencar Queiroz, Lorena E. P. Salomão.

## Data

The eICU-CRD v2.0 is available on PhysioNet to credentialed users (https://physionet.org/content/eicu-crd/2.0/).
In this study, only Lucas Alencar Queiroz, the author with credentialed access, accessed and analyzed the data.
This repository contains no data.

## Running

Python 3.10 with the packages in requirements.txt, and R 4.2 with givitiR and jsonlite.

    pip install -r requirements.txt
    export EICU_DIR=/path/to/eicu-crd/2.0
    python run_all.py

If Rscript is not on the PATH, set RSCRIPT to its location. Results go to output/.

The scripts run in numerical order: cohort selection (01, 02), first 24 hours of ICU data (03), scores (04),
discrimination and calibration (05, 06), cut-offs (07), patient characteristics (08) and figures (09).

APACHE IVa predictions come from the database. SAPS II and OASIS were calculated as in the MIT-LCP mimic-code, with
missing values scored as normal. Definitions are given in the article.

## License

MIT
