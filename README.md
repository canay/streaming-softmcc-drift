# MemSoftMCC replication package

Code and saved evidence for **MemSoftMCC: Memory-Weighted SoftMCC Estimators for
Prequential Monitoring under Concept Drift**.

This repository separates the earlier descriptive drift diagnostics from the
prospectively specified matched-memory, finite-bias, dependence, and covariance-
support studies. It includes negative findings and admissibility failures.

- `code/`, `results/`, `figures/` and `docs/` contain the earlier diagnostics.
- `studies/` contains the four later study layers, configurations and saved
  replication-unit/aggregate results.
- `protocols/` contains public-safe copies of prospective protocols.
- `PUBLIC_COPY_PROVENANCE.json` binds original and public-copy identities.
- `SHA256SUMS.txt` identifies files actually shipped in this snapshot.

Start with the read-only consistency check:

```text
python verify_saved_results.py
```

The full requirements, commands, expected outputs, numerical conventions and
reproduction limits are in [`STUDIES.md`](STUDIES.md). The root requirements
belong to the legacy run; the later numerical stacks are recorded separately.
The September runners require Linux and NumPy 2.4.6. A full rerun is distinct
from inspecting the saved summaries and can take substantially longer.

Raw third-party datasets and prepared caches are not distributed. New-study
sample-level trajectory NPZs are also excluded; the package supplies their
original identities, reproduction code and derived results. The previously
released NPZ files under `results/` contain monitoring traces, not raw feature
matrices. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for access instructions.
The original IoTID20 cache's row-selection provenance is incomplete, so its
recorded hash does not guarantee reconstruction by the current builder.

The repository is available at
https://github.com/canay/streaming-softmcc-drift. It is an experiment replication
repository; manuscript sources, publisher PDFs and internal review/state files
are excluded. The software is distributed under the MIT License. Third-party
data remain subject to their original terms. No release tag or DOI is implied.
