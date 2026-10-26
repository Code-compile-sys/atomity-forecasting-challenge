# Phase 2 — Training-only EDA notes

## Scope and leakage control

The analysis uses `build_targets()` to form the aggregate time series, then applies the required chronological 24-timestamp holdout split. All reported EDA statistics below use the first 110 target timestamps and the VM-level readings at those training timestamps only. Holdout target values are not summarized, charted, or used to draw conclusions. Metadata CPU summary fields (`max_cpu`, `avg_cpu`, and `p95_max_cpu`) are excluded from all segment analyses.

## Data coverage and quality

- The training partition contains 110 five-minute timestamps, from raw timestamp 0 through 32,700 seconds (UTC 00:00 through 09:05 in the source-relative time representation).
- The observed training cadence is exactly 300 seconds, with no gaps in the target timestamp sequence.
- The training readings contain 1,100,000 VM-level records and represent 10,000 distinct VMs.
- There are no missing timestamps, VM IDs, or `avg_cpu` values among those training readings, and no duplicate `(timestamp, vm_id)` pairs.
- Every observed training `avg_cpu` value is within the documented 0–100 percent range.
- The source timestamps are seconds relative to the trace, not real-world wall-clock time. The derived UTC dates therefore should not be interpreted as the actual date/time of the workload.

### Timestamp quality (training partition only)

**Cadence**
Value: 300 seconds
So what: Data is collected every 5 minutes and appears suitable for fixed-interval forecasting.

**Number of gaps**
Value: 0
So what: No adjacent training timestamps exceed the typical 300-second interval.

**Number of duplicate timestamps**
Value: 0
So what: Each aggregated training timestamp is unique; repeated raw VM rows at a timestamp are expected and are not counted as duplicate time-series points.

## VM lifetimes and churn (training partition only)

Lifetime is measured as elapsed seconds between a VM's first and last observed training readings. This is an observed window, not necessarily the VM's full provisioned lifetime. The start boundary is left-censored (VMs could have existed earlier) and the end boundary is right-censored (VMs may persist later), so boundary first/last appearances are not counted as confirmed arrivals/departures.

**Median observed VM lifetime**
Metric: 32,700 seconds (9.08 hours)
So what: The typical VM is observed throughout the available training window; its true lifetime may be longer.

**Mean observed VM lifetime**
Metric: 32,700 seconds (9.08 hours)
So what: All 10,000 VMs have the same first-to-last observed span in training.

**VM arrivals within the observed training window**
Metric: 0
So what: No VMs first appear after the left-censored start boundary.

**VM departures within the observed training window**
Metric: 0
So what: No VMs disappear before the right-censored end boundary.

All 10,000 VMs are first seen at the training start and last seen at its end. These boundary counts are censored observations, not evidence that 10,000 VMs were actually created/deleted then. This slice shows no measurable internal churn, but cannot establish VM lifetimes or behavior outside the training window.

## Per-VM CPU utilization (training partition only)

For each VM, mean CPU is computed from its `avg_cpu` readings at training timestamps only.

**VMs with mean CPU <5%**
4,635 VMs (46.35%)
That means : Nearly half of the VMs average below 5% CPU, indicating substantial low-utilization capacity in this trace.

**VMs with mean CPU >50%**
220 VMs (2.20%)
That means : Only a small share sustain mean CPU above 50%; this does not rule out short-lived utilization spikes.

## Aggregate CPU series (training partition only)

The notebook plots aggregate CPU at each five-minute training timestamp with a trailing 12-observation (one-hour) rolling mean. The rolling window uses only the current and previous observations.

- **Trend:** the trailing mean decreases from 9.411% at its first complete window to 9.216% at the final training timestamp. This suggests a modest decline across the observed training period, not a long-term trend.
- **Volatility:** the target standard deviation is 0.194 percentage points, with values ranging from 9.112% to 9.912%. The aggregate is comparatively stable despite the much wider VM-level CPU distribution.
- **Seasonality:** no daily/weekly pattern can be established from approximately nine hours of observations. Short-term local variation is visible but is not evidence of recurring seasonality.
- **Spikes/outliers:** the maximum training value is 9.912% at raw timestamp 16,500 seconds. This is a local peak; it is not labeled an anomaly without a separate, pre-specified anomaly rule.

## Temporal utilization patterns (training partition only)

The 110 training observations span only one source-relative UTC date, from 00:00 to 09:05. Since raw timestamps are seconds relative to trace start, the converted UTC weekday/hour are not actual workload wall-clock time. Hour means below are grouped by elapsed hours from trace start.

### Mean CPU by elapsed hour

| Elapsed hour | Mean aggregate CPU | Observations |
|---:|---:|---:|
| 0 | 9.411% | 12 |
| 1 | 9.405% | 12 |
| 2 | 9.367% | 12 |
| 3 | 9.361% | 12 |
| 4 | 9.615% | 12 |
| 5 | 9.707% | 12 |
| 6 | 9.372% | 12 |
| 7 | 9.295% | 12 |
| 8 | 9.239% | 12 |
| 9 | 9.138% | 2 |

**Value:** The peak hourly mean is 9.707% in elapsed hour 5 (12 observations); the lowest is 9.138% in elapsed hour 9 (only 2 observations).
**So what:** CPU is higher around hours 4–5 and falls later in the observed window, but the final partial-hour minimum is based on only two samples; use these as short-window monitoring signals, not a clock-based schedule.

### Day-of-week support

**Value:** 1 source-relative weekday represented (all 110 observations map to Thursday UTC).
**So what:** Day-of-week comparisons are unsupported; more than one real calendar week of correctly anchored timestamps would be required.

### Repeating utilization cycles

**Value:** Lag-1-hour autocorrelation is 0.367; lag-2-hour autocorrelation is -0.026.
**So what:** These exploratory correlations do not establish a repeating cycle; the short training window has insufficient repeated periods to justify seasonal forecasts.

## Workload shape

- `active_vm_count` is constant at 10,000 for every training timestamp. It provides no variation to forecast within this window; any forecast should respect the count’s nonnegative interpretation and should not claim evidence of churn from this segment.
- Aggregate CPU utilization averages 9.414%, with a standard deviation of 0.194 percentage points and a training range of 9.112%–9.912%.
- By contrast, the distribution of the 1.1 million VM-level readings is much wider: mean 9.414%, median 5.120%, 75th percentile 9.648%, 95th percentile 37.293%, and maximum 99.488%. This right-skew and spread show that the aggregate mean conceals substantial VM-level heterogeneity.
- Training-period CPU peaks occur around raw timestamps 16,500–20,400 seconds; the maximum aggregate CPU value is approximately 9.912%. The smallest value is approximately 9.112% at raw timestamp 32,400 seconds. These are local fluctuations, not evidence of recurring daily cycles.

## Segment differences

Segmentation uses the upstream metadata buckets as supplied; no new cutoffs or derived `core_bucket` / `memory_bucket` columns were needed:

- `vm_category`: Azure workload category (`Delay-insensitive`, `Interactive`, `Unknown`).
- `virtual_core_bucket`: supplied virtual-core count buckets (`2`, `4`, `8`, `24`, `>24`).
- `memory_gb_bucket`: supplied memory-in-GB buckets (`2`, `4`, `8`, `32`, `64`, `>64`).

These fields are categorical bucket labels; `>24` and `>64` remain open-ended labels, not numeric values for modeling. CPU-summary metadata columns are excluded from CPU segment summaries.

For each dimension, average CPU is the mean of VM-level `avg_cpu` readings at training timestamps. Aggregate utilization contribution is defined as `100 × segment sum(avg_cpu) / sum(avg_cpu across all training VM readings)`. This is the share of summed CPU utilization readings; it is not a share of VM capacity, since core counts/capacity are not available as exact values. Per-VM reading coverage is balanced for this training slice.

### Workload category

| Category | VM count | Average CPU | Aggregate utilization contribution |
|---|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.257% | 54.206% |
| Interactive | 3,091 | 11.505% | 37.776% |
| Unknown | 729 | 10.355% | 8.018% |

**Observation:** Interactive VMs average 11.505% CPU, higher than Delay-insensitive VMs at 8.257%, while Delay-insensitive VMs contribute the largest total share (54.206%) due to their larger population.
**So what:** Capacity planning should consider both workload intensity and segment population; per-category forecasts could distinguish high per-VM demand from high total contribution.

### Virtual-core bucket

| Core bucket | VM count | Average CPU | Aggregate utilization contribution |
|---|---:|---:|---:|
| 2 | 7,534 | 9.764% | 78.142% |
| 4 | 1,908 | 7.919% | 16.049% |
| 8 | 486 | 9.658% | 4.986% |
| 24 | 62 | 11.429% | 0.753% |
| >24 | 10 | 6.619% | 0.070% |

**Observation:** The 2-core bucket accounts for 78.142% of the summed CPU readings, and the 24-core bucket has the highest mean (11.429%) but only 62 VMs.
**So what:** The common 2-core segment dominates aggregate behavior, while high averages in small segments need uncertainty-aware interpretation due to low sample counts.

### Memory bucket

| Memory bucket (GB) | VM count | Average CPU | Aggregate utilization contribution |
|---|---:|---:|---:|
| 2 | 3,904 | 8.802% | 36.503% |
| 4 | 2,475 | 11.460% | 30.128% |
| 8 | 2,146 | 9.143% | 20.843% |
| 32 | 1,383 | 7.814% | 11.479% |
| 64 | 82 | 11.217% | 0.977% |
| >64 | 10 | 6.619% | 0.070% |

**Observation:** 2-GB VMs contribute the largest share (36.503%) by population, while 4-GB VMs have the highest mean CPU (11.460%) among the larger groups.
**So what:** Aggregated demand is shared across the common small-memory segments; segment averages alone do not identify total capacity demand.

These are descriptive statistics from the training window, not causal effects or validated forecasts. Small buckets (especially 24-core, >24-core, 64-GB, and >64-GB) have low sample counts, so their averages may be unstable.

## Modeling implications and caveats

- With only about 9 hours of training history and five-minute sampling, the data cannot support claims about daily or weekly seasonality. Calendar seasonal features would be weakly justified for this source-relative trace.
- Since the active count is flat in training, a simple persistence forecast is an important baseline for that target. CPU utilization changes modestly at the aggregate level, while individual VM readings vary considerably.
- Segment-level CPU differences may motivate diagnostics, but using metadata CPU-summary fields would leak precomputed target-like information and is prohibited. Only segment labels and training-period readings were used here.
- EDA and all model/hyperparameter choices must remain inside the 110-timestamp training partition. Use rolling-origin validation there; reserve the final 24 timestamps for the required final evaluation only.

## Rolling-origin baseline evaluation (training partition only)

Baselines were evaluated with expanding chronological folds on the 110 training timestamps. Rows were ordered by the UTC `datetime` column. Each fold starts with 50 training observations, validates on 24 observations, then advances the forecast origin by 6 observations, for seven overlapping folds. The final 24 challenge holdout timestamps were not included in fitting, validation, or model selection. Metrics below are arithmetic means across folds.

The notebook's `baseline_cv_results` table stores one row for each target, baseline, and fold (70 rows total), including fold number, baseline name, fold boundaries and sizes, MAE, RMSE, normalized MAE, sMAPE, and count of scored observations.

### Average metrics and ranking

Metrics are arithmetic averages over the seven folds. Baselines are ranked within each target by mean MAE (lower is better); equal scores share a rank.

| Target | Rank | Baseline | Mean MAE | Mean RMSE | Mean normalized MAE | Mean sMAPE |
|---|---:|---|---:|---:|---:|---:|
| `active_vm_count` | 1 (tie) | Persistence | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | 1 (tie) | Last6Mean | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | 1 (tie) | Last12Mean | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | 1 (tie) | TrainingMean | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | 1 (tie) | SeasonalNaive | 0.000 | 0.000 | 0.000 | 0.000% |
| `aggregate_cpu_utilization` | 1 | TrainingMean | 0.209 | 0.238 | 1.154 | 2.207% |
| `aggregate_cpu_utilization` | 2 | Persistence | 0.215 | 0.251 | 1.000 | 2.261% |
| `aggregate_cpu_utilization` | 3 | Last6Mean | 0.243 | 0.272 | 1.211 | 2.555% |
| `aggregate_cpu_utilization` | 4 | Last12Mean | 0.260 | 0.288 | 1.389 | 2.732% |
| `aggregate_cpu_utilization` | 5 | SeasonalNaive | 0.272 | 0.306 | 1.423 | 2.852% |

**Observation:** TrainingMean ranks first for `aggregate_cpu_utilization` by mean MAE (0.209 percentage points) and has the lowest mean RMSE and sMAPE. All five baselines tie for `active_vm_count` because the target is constant.

**So what:** Use TrainingMean as the leading simple CPU benchmark for this validation design, while retaining persistence as the relative-MAE reference; the flat active count provides no basis to distinguish baselines or establish churn performance.

| Target | Baseline | Mean MAE | Mean RMSE | Normalized MAE | Mean sMAPE |
|---|---|---:|---:|---:|---:|
| `active_vm_count` | Persistence | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | Last-6 mean | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | Last-12 mean | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | Training mean | 0.000 | 0.000 | 0.000 | 0.000% |
| `active_vm_count` | Seasonal naive (12) | 0.000 | 0.000 | 0.000 | 0.000% |
| `aggregate_cpu_utilization` | Persistence | 0.215 | 0.251 | 1.000 | 2.261% |
| `aggregate_cpu_utilization` | Last-6 mean | 0.243 | 0.272 | 1.211 | 2.555% |
| `aggregate_cpu_utilization` | Last-12 mean | 0.260 | 0.288 | 1.389 | 2.732% |
| `aggregate_cpu_utilization` | Training mean | 0.209 | 0.238 | 1.154 | 2.207% |
| `aggregate_cpu_utilization` | Seasonal naive (12) | 0.272 | 0.306 | 1.423 | 2.852% |

CPU errors are in percentage points; sMAPE is a percentage. Normalized MAE is model MAE divided by persistence MAE within each fold, then averaged over folds. Because this is an average of fold-level ratios, it can rank methods differently from mean MAE. For the constant active-count series, both forecast and persistence have zero MAE, represented as normalized MAE 0.

**Observation:** All baselines have zero error for `active_vm_count` across the seven folds because this training series is constant at 10,000 VMs.

**So what:** The perfect scores follow from a flat target and do not demonstrate that these methods can anticipate future VM churn.

**Observation:** The training-mean baseline has the lowest mean aggregate CPU MAE (0.209 percentage points) and RMSE (0.238 percentage points); persistence has MAE 0.215 and RMSE 0.251.

**So what:** Averaging all available training history is a competitive simple benchmark for this short, relatively stable CPU series; comparisons to other models must use the same folds.

## Baseline forecast definitions

The baseline functions in `src/cwfc/baselines.py` return a forecast series aligned to the supplied forecast template:

- **Persistence:** repeats the latest observed training value.
- **Last-6 mean:** repeats the mean of the latest six training observations (30 minutes at the five-minute cadence).
- **Last-12 mean:** repeats the mean of the latest twelve training observations (one hour).
- **Training mean:** repeats the mean of all observations available in the current training fold.
- **Seasonal naive:** repeats values from the latest observed cycle. The default season length is **12 observations (one hour)**, matching the five-minute cadence.

**Observation:** The trace is too short to establish a repeating one-hour seasonal pattern, and earlier temporal analysis did not demonstrate reliable seasonality.

**So what:** The 12-observation seasonal-naive forecast is included as a transparent candidate benchmark, not as a claim of proven seasonality; in this training-only comparison, it performs worse than persistence and the training mean for aggregate CPU.

### Configured fold metadata

For rolling-origin CV, the training partition has 110 rows. The configured minimum training size is 50, validation horizon is 24 rows, and origin step is 6 rows. Fold row bounds below are zero-based with exclusive end indices; all validation windows overlap by 18 observations. Timestamps are ordered using `datetime`.

| Fold | Train rows `[start:end)` | Validation rows `[start:end)` | Train size | Validation size |
|---:|---|---|---:|---:|
| 1 | `[0:50)` | `[50:74)` | 50 | 24 |
| 2 | `[0:56)` | `[56:80)` | 56 | 24 |
| 3 | `[0:62)` | `[62:86)` | 62 | 24 |
| 4 | `[0:68)` | `[68:92)` | 68 | 24 |
| 5 | `[0:74)` | `[74:98)` | 74 | 24 |
| 6 | `[0:80)` | `[80:104)` | 80 | 24 |
| 7 | `[0:86)` | `[86:110)` | 86 | 24 |

**Observation:** Seven folds are generated, and each validation window starts six observations after the prior fold's origin.

**So what:** Overlapping validation windows provide several near-term forecast origins from the small training history, while the final 24 challenge holdout observations remain excluded.
