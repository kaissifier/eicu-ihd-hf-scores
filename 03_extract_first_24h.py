import polars as pl

from common import EXTRACT_DIR, out, table

ids = pl.read_parquet(out("population.parquet"))["patientunitstayid"].to_list()
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

TIMED = {
    "vitalPeriodic": ("observationoffset", 0,
                      ["temperature", "heartrate", "respiration", "systemicsystolic", "systemicmean"]),
    "vitalAperiodic": ("observationoffset", 0, ["noninvasivesystolic", "noninvasivemean"]),
    "lab": ("labresultoffset", -360, ["labname", "labresult"]),
    "nurseCharting": ("nursingchartoffset", 0, ["nursingchartcelltypevalname", "nursingchartvalue"]),
    "respiratoryCharting": ("respchartoffset", 0, ["respchartvaluelabel", "respchartvalue"]),
    "infusionDrug": ("infusionoffset", 0, ["drugname", "drugrate", "patientweight"]),
    "intakeOutput": ("intakeoutputoffset", 0, ["cellpath", "celllabel", "cellvaluenumeric"]),
}

for name, (offset, start, cols) in TIMED.items():
    df = (pl.scan_csv(table(name), infer_schema_length=0)
          .select(["patientunitstayid", offset, *cols])
          .with_columns(pl.col("patientunitstayid").cast(pl.Int64), pl.col(offset).cast(pl.Float64, strict=False))
          .filter(pl.col("patientunitstayid").is_in(ids) & pl.col(offset).is_between(start, 1440))
          .collect(engine="streaming"))
    df.write_parquet(EXTRACT_DIR / f"{name}.parquet")
    print(f"{name}: {df.height:,} rows")

for name in ("apacheApsVar", "apachePredVar", "patient"):
    df = pl.read_csv(table(name), infer_schema_length=0).with_columns(pl.col("patientunitstayid").cast(pl.Int64))
    df.filter(pl.col("patientunitstayid").is_in(ids)).write_parquet(EXTRACT_DIR / f"{name}.parquet")
    print(f"{name}: done")
