import numpy as np
import pandas as pd

from common import out, table

cohort = pd.read_parquet(out("cohort.parquet"))

dx = pd.read_csv(table("diagnosis"), usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
dx = dx[dx.patientunitstayid.isin(cohort.patientunitstayid)]
text = dx.diagnosisstring.fillna("").str.lower()
acute_hf = dx[text.str.contains("heart failure|chf") & text.str.contains("acute|decompensated")]
by_diagnosis = set(acute_hf.loc[acute_hf.diagnosisoffset <= 1440, "patientunitstayid"])

admission_dx = cohort.apacheadmissiondx.fillna("").str.lower()
by_admission = set(cohort.loc[admission_dx.str.contains("chf|congestive heart failure"), "patientunitstayid"])

apache = pd.read_csv(table("apachePatientResult"),
                     usecols=["patientunitstayid", "apacheversion", "predictedhospitalmortality"])
apache = apache[apache.apacheversion == "IVa"].drop_duplicates("patientunitstayid")
apache["p_apache"] = pd.to_numeric(apache.predictedhospitalmortality, errors="coerce")
apache.loc[apache.p_apache <= 0, "p_apache"] = np.nan  # -1 flags stays without a valid prediction

pop = cohort[cohort.patientunitstayid.isin(by_diagnosis | by_admission)]
pop = pop.merge(apache[["patientunitstayid", "p_apache"]], on="patientunitstayid", how="left")
pop["criterion"] = np.where(pop.patientunitstayid.isin(by_admission), "admission", "diagnosis_24h")

cols = ["patientunitstayid", "hospitalid", "died_hospital", "criterion", "p_apache", "hospitaldischargestatus"]
pop[cols].to_parquet(out("population.parquet"), index=False)
print(f"{len(pop)} patients, {pop.hospitalid.nunique()} hospitals "
      f"(HF admission {int((pop.criterion == 'admission').sum())}, HF diagnosis within 24 h "
      f"{int((pop.criterion == 'diagnosis_24h').sum())})")
