import json
import re

import numpy as np
import pandas as pd
import polars as pl

from common import out, table

IHD_DOTTED = re.compile(r"^(410\.|411\.|412|413\.|414\.|I2[0-5])")
IHD_PLAIN = re.compile(r"^(410|411|412|413|414|I2[0-5])")

# A condition is present if an ICD code is in the diagnosis table or pastHistory has a matching entry.
ICD = {
    "diabetes": r"^250|^E1[0-4]",
    "ckd": r"^585|^N18",
    "stroke": r"^43[0-8]|^I6[0-9]",
    "htn": r"^40[1-5]|^I1[0-5]",
    "copd": r"^49[0-6]|^J4[0-4]",
    "pvd": r"^44[0-4]|^I7[0-4]",
    "afib": r"^4273|^I48",
}
PAST_HISTORY = {
    "diabetes": r"diabetes",
    "ckd": r"renal failure|chronic kidney|esrd|ckd",
    "stroke": r"stroke|cva|cerebrovascular",
    "htn": r"hypertension",
    "copd": r"copd|chronic obstructive",
    "pvd": r"peripheral vascular|pvd",
    "afib": r"atrial fibrillation|atrial flutter",
}


def has_ihd(codes):
    if pd.isna(codes):
        return False
    return any(IHD_DOTTED.match(c.strip()) or IHD_PLAIN.match(c.strip().replace(".", ""))
               for c in str(codes).split(","))


dx = pd.read_csv(table("diagnosis"), usecols=["patientunitstayid", "icd9code"], dtype={"icd9code": "string"})
ihd = set(dx.loc[dx.icd9code.map(has_ihd), "patientunitstayid"])

pat = pd.read_csv(table("patient"))
flow = {"icu_stays": len(pat), "hospitals": int(pat.hospitalid.nunique())}

c = pat[pat.patientunitstayid.isin(ihd)].copy()
flow["ihd_stays"] = len(c)

age = c.age.astype(str).str.strip().replace({"> 89": "90", "": np.nan, "nan": np.nan})
c["age_num"] = pd.to_numeric(age, errors="coerce")
flow["excluded_age"] = int((~(c.age_num >= 18)).sum())
c = c[c.age_num >= 18]

n = len(c)
c = c.sort_values(["patienthealthsystemstayid", "unitvisitnumber"]).drop_duplicates("patienthealthsystemstayid")
flow["excluded_readmission"] = n - len(c)

n = len(c)
c["icu_los_h"] = c.unitdischargeoffset / 60
c = c[c.icu_los_h >= 4].copy()
flow["excluded_icu_under_4h"] = n - len(c)
flow["cohort"] = len(c)
flow["cohort_hospitals"] = int(c.hospitalid.nunique())

# a missing discharge status counts as survival here; the performance analysis excludes these patients
c["died_hospital"] = (c.hospitaldischargestatus == "Expired").astype(int)

ids = c.patientunitstayid.tolist()
codes = (pl.scan_csv(table("diagnosis"), schema_overrides={"icd9code": pl.Utf8})
         .filter(pl.col("patientunitstayid").is_in(ids))
         .select(["patientunitstayid", "icd9code"])
         .collect().to_pandas())
codes["icd9code"] = codes.icd9code.fillna("").astype(str)
history = (pl.scan_csv(table("pastHistory"), schema_overrides={"pasthistorypath": pl.Utf8, "pasthistoryvalue": pl.Utf8})
           .filter(pl.col("patientunitstayid").is_in(ids))
           .select(["patientunitstayid", "pasthistorypath"])
           .collect().to_pandas())
history_path = history.pasthistorypath.fillna("").str.lower()

for name, pattern in ICD.items():
    rx = re.compile(pattern)
    coded = codes.icd9code.map(lambda s: any(rx.match(x.replace(".", "").strip()) for x in s.split(",")))
    in_history = history_path.str.contains(PAST_HISTORY[name], regex=True)
    with_condition = set(codes.loc[coded, "patientunitstayid"]) | set(history.loc[in_history, "patientunitstayid"])
    c[name] = c.patientunitstayid.isin(with_condition).astype(int)

cols = ["patientunitstayid", "patienthealthsystemstayid", "hospitalid", "age_num", "gender", "unitadmitsource",
        "apacheadmissiondx", "hospitaldischargestatus", "icu_los_h", "died_hospital", *ICD]
c[cols].to_parquet(out("cohort.parquet"), index=False)
with open(out("flow.json"), "w") as f:
    json.dump(flow, f, indent=1)
print(flow)
