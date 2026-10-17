suppressPackageStartupMessages({
  library(givitiR)
  library(jsonlite)
})

out_dir <- Sys.getenv("OUT_DIR", "output")
d <- read.csv(file.path(out_dir, "predictions.csv"))

belts <- list()
for (s in c("p_apache", "sapsii_prob", "oasis_prob")) {
  cb <- givitiCalibrationBelt(o = d$death, e = d[[s]], devel = "external", confLevels = c(0.80, 0.95), maxDeg = 4)
  belts[[s]] <- list(
    n = cb$n, m = cb$m, statistic = cb$statistic, p_value = cb$p.value, seqP = cb$seqP,
    bounds = lapply(cb$cbBoundByConfLevel, function(b) list(L = b$L, U = b$U)),
    inters = lapply(cb$intersByConfLevel, function(z) list(over = z$overBisector, under = z$underBisector))
  )
  cat(s, ": polynomial degree", cb$m, ", p =", format(cb$p.value, digits = 3), "\n")
}
write_json(belts, file.path(out_dir, "calibration_belt.json"), auto_unbox = TRUE, digits = NA)
