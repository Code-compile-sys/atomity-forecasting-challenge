# Cloud Workload Forecasting Challenge

This repository builds leakage-safe forecasts for the required Azure VM workload targets: `active_vm_count` and `aggregate_cpu_utilization`. Raw timestamps are retained in the output; UTC datetimes are used only for time-series modeling.

## Requirements and installation

Use Python 3.11 or newer. From the repository root, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the runtime libraries used by the source and notebooks, including pandas, NumPy, scikit-learn, statsmodels, matplotlib, Jupyter, and ipykernel.

## Dataset

Download the fixed [`data-v1` release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract `data-v1.tar.gz` into the repository root as `challenge_data/`. It should contain:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify the release checksums before running the pipeline. On macOS/Linux:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
cd ..
```

On Windows PowerShell, from the repository root:

```powershell
Get-Content .\challenge_data\SHA256SUMS | ForEach-Object {
    if ($_ -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') { throw "Invalid checksum entry: $_" }
    $expected = $Matches[1].ToLower()
    $name = $Matches[2].Trim()
    $actual = (Get-FileHash (Join-Path .\challenge_data $name) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) { throw "Checksum mismatch: $name" }
    "$name`: OK"
}
```

The dataset is distributed as a release asset and is intentionally not committed to this repository.

## Generate predictions

From the repository root, run:

```bash
python -m cwfc.predict --data challenge_data --output predictions/predictions.csv
```

The default holdout is the final 24 timestamps. The generated CSV contains exactly `timestamp,active_vm_count,aggregate_cpu_utilization`, in chronological order, without an index. Predictions are validated for matching timestamps, finite values, non-negative VM counts, and CPU utilization in `[0,100]`. To use another horizon explicitly, pass `--holdout N`.

## Run tests

```bash
python -m pytest -q
```

The tests include output-contract checks, a tiny synthetic end-to-end dataset, and a leakage check that changes holdout targets and verifies the forecasts remain unchanged.

## Modeling decisions

- **Targets and leakage control:** Targets are aggregated from VM-level readings. The final holdout is separated before model selection or fitting; its target values are excluded from CV, selection, and forecast calls. Metadata CPU-summary columns are not model features.
- **Candidates:** exactly three forecasters are compared: `Last12Mean` (the moving-average baseline), Ridge using lagged target values, and damped exponential smoothing (ETS).
- **Validation:** expanding rolling-origin cross-validation uses only the training partition, with a 50-point minimum training window, 24-step forecast horizon, and 6-point origin step. Each candidate is scored by fold MAE and RMSE.
- **Selection:** rank by mean CV MAE. Ridge or DampedETS is selected only when it improves on `Last12Mean` by at least 3%; otherwise the baseline wins. When the two model candidates tie, prefer DampedETS.
- **Intervals:** the selected candidate's residuals (actual minus forecast) are pooled across CV folds. Their 10th and 90th percentiles define the 80% interval; bounds are clipped to the valid target ranges.
- **Capacity policy:** CV compares exactly three CPU provisioning policies: forecast alone, forecast plus half the upper interval margin, and forecast plus the full margin. The full-margin policy reduces shortfall risk at the cost of more unused headroom; `notes.md` records the CV comparison and recommendation.

The final holdout scores and one-time forecast table are documented in `notes.md`; they are not used to retune the model or policy.
