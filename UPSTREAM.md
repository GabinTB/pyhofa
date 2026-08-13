# Upstream snapshot

This Python port was prepared against:

- repository: `GuanglinHuang/hofa`
- branch: `master`
- upstream commit: `493645e27f06c021d9474412aeef98a66a1468f5`
- upstream R package version: `0.9.4`

Files explicitly reviewed for the port include:

- `R/functions.R`
- `R/M2.select.R`, `R/M3.select.R`, `R/M4.select.R`
- `R/M2.pca.R`, `R/M2.mle.R`, `R/M2.gmm.R`
- `R/M3.gmm.R`, `R/M3.als.R`, `R/M4.als.R`
- `R/Adaptive.HFA.R`
- `R/Portfolio.IC.R`, `R/Portfolio.PC.R`
- `R/hofa.sim.R`
- `src/MLE_BL_cpp.cpp`, `src/MC4sample.cpp`
- the upstream `testthat` suite

See the README section "Intentional corrections and Python-specific choices" for places where the source code and its apparent mathematical intent differ.
