#  Training-only EDA notes

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

## Focused three-candidate CV

The current evaluator compares exactly the last-12 moving mean, Ridge, and Damped ETS for any requested target column. It receives only the training partition, orders each fold by `datetime`, and returns per-fold plus mean MAE and RMSE; normalized MAE and sMAPE remain guarded in the metric helpers but are not added to this comparison output.

| Target | Model | Mean MAE | Mean RMSE |
|---|---|---:|---:|
| `active_vm_count` | Last12Mean | 0.000 | 0.000 |
| `active_vm_count` | Ridge | 0.000 | 0.000 |
| `active_vm_count` | DampedETS | 0.000 | 0.000 |
| `aggregate_cpu_utilization` | Last12Mean | 0.260 | 0.288 |
| `aggregate_cpu_utilization` | Ridge | 0.224 | 0.255 |
| `aggregate_cpu_utilization` | DampedETS | 0.244 | 0.275 |

Each target has seven rolling-origin folds (50 initial observations, 24-step horizon, 6-step advance). All three methods tie on the constant active-count target; Ridge has the lowest mean MAE and RMSE among these candidates for aggregate CPU utilization. No final-holdout target values enter evaluation.

Fold boundaries are ordered by `datetime`; each fold's validation window is 24 points, with the origin advancing six points:

| Fold | Train rows `[start:end)` | Validation rows `[start:end)` |
|---:|---|---|
| 1 | `[0:50)` | `[50:74)` |
| 2 | `[0:56)` | `[56:80)` |
| 3 | `[0:62)` | `[62:86)` |
| 4 | `[0:68)` | `[68:92)` |
| 5 | `[0:74)` | `[74:98)` |
| 6 | `[0:80)` | `[80:104)` |
| 7 | `[0:86)` | `[86:110)` |

## Optional 80% prediction intervals

`forecast_targets_with_intervals()` selects the winner separately for both targets using training-only CV, pools that model's actual-minus-forecast residuals across all folds and horizon steps, and offsets its final forecast by the pooled 10th/90th residual quantiles. It clips VM counts at zero and CPU utilization to `[0,100]`; only holdout timestamps are supplied, and the interval table is separate from the required point-prediction CSV.

## Aggregate CPU error analysis (training-only CV)

The selected model was Ridge. The peak cutoff is the training-series 90th percentile, **9.6684% CPU**. The seven 24-step folds contribute 168 forecast cases; because folds overlap, cases are repeated across forecast origins.

| Point group | Forecast cases | MAE | RMSE | Bias (actual − forecast) |
|---|---:|---:|---:|---:|
| Peak (`actual > 9.6684%`) | 27 | 0.4104 | 0.4611 | +0.4104 |
| Non-peak | 141 | 0.1885 | 0.2279 | +0.1400 |

**So what:** Peak errors are more than twice as large, and positive bias in both groups means Ridge tends to under-forecast, especially during high CPU periods.

### Fold MAE ranking

| Rank | Fold | MAE |
|---:|---:|---:|
| 1 | 1 | 0.4148 |
| 2 | 2 | 0.3296 |
| 3 | 3 | 0.2523 |
| 4 | 6 | 0.1655 |
| 5 | 5 | 0.1545 |
| 6 | 7 | 0.1446 |
| 7 | 4 | 0.1076 |

**Fold 1 likely reason:** The actual CPU rises into a sustained high period (9.91% peak) while Ridge remains near 9.2–9.4%, indicating under-response to the rise.

**Fold 2 likely reason:** Ridge recursively drifts downward while actual CPU remains elevated, then the actual series drops late in the window; the forecast misses both the level and transition.

**So what:** Most error is concentrated in early validation origins where the recent training history poorly represents subsequent CPU levels and changes.

### Optional MAE by horizon step

Across folds, the lowest horizon-step MAE was **0.1092 at step 2** and the highest was **0.3798 at step 16**.

**So what:** Error varies irregularly by forecast step rather than increasing monotonically, so a simple horizon-dependent error assumption is not supported by this short trace.

## Aggregate CPU provisioning policy comparison (training-only CV)

The selected model is Ridge. The upper 80% interval margin is the pooled 90th-percentile of its CV residuals (actual minus forecast), **0.4984 percentage points**. The table scores all 168 CV forecast cases; the final holdout is excluded.

| Policy | Under-provisioning rate | Average shortfall | Wasted headroom |
|---|---:|---:|---:|
| A: forecast | 82.14% | 0.2481 | 0.1139 |
| B: forecast + half upper-interval margin | 32.74% | 0.1679 | 0.1795 |
| C: forecast + full upper-interval margin | 10.12% | 0.0868 | 0.3601 |

Under-provisioning rate is the share of all points where actual exceeds provision; average shortfall averages actual minus provision on those cases. Wasted headroom averages provision minus actual only where provision meets or exceeds actual. Values are CPU percentage points except the rate.

**Recommendation:** Use Policy C for capacity planning when avoiding CPU shortfalls is the priority: it has the lowest under-provisioning rate (10.12%) and average shortfall (0.0868) on CV. Its trade-off is the largest wasted headroom (0.3601 percentage points), so use Policy B if the operational cost of spare capacity outweighs the increased shortfall risk.

## Final holdout check (run once)

The selected models were fit on the 110 training points only and forecast the final 24 timestamps. Interval bounds use the selected models' pooled training-CV residuals. The holdout actuals were used only for the final scores below, not for model or policy selection.

| Target | Selected model | Selected MAE | Selected RMSE | Baseline | Baseline MAE | Baseline RMSE |
|---|---|---:|---:|---|---:|---:|
| `active_vm_count` | Last12Mean | 0.0000 | 0.0000 | Last12Mean | 0.0000 | 0.0000 |
| `aggregate_cpu_utilization` | Ridge | 0.3012 | 0.3604 | Last12Mean | 0.1230 | 0.1613 |

Policy C (full upper 80% interval margin) under-provisioned **25.00%** of CPU holdout points (6 of 24). Average wasted headroom was **0.3151 percentage points**, calculated on the 18 points where provision met or exceeded actual.

| Raw timestamp (seconds) | Active VM forecast | Active VM lower 80% | Active VM upper 80% | CPU forecast | CPU lower 80% | CPU upper 80% |
|---:|---:|---:|---:|---:|---:|---:|
| 33000 | 10000 | 10000 | 10000 | 9.176208 | 9.113860 | 9.674581 |
| 33300 | 10000 | 10000 | 10000 | 9.139659 | 9.077310 | 9.638031 |
| 33600 | 10000 | 10000 | 10000 | 9.093719 | 9.031370 | 9.592091 |
| 33900 | 10000 | 10000 | 10000 | 9.100103 | 9.037754 | 9.598475 |
| 34200 | 10000 | 10000 | 10000 | 9.074400 | 9.012051 | 9.572772 |
| 34500 | 10000 | 10000 | 10000 | 9.068160 | 9.005811 | 9.566532 |
| 34800 | 10000 | 10000 | 10000 | 9.059044 | 8.996695 | 9.557416 |
| 35100 | 10000 | 10000 | 10000 | 9.025669 | 8.963320 | 9.524041 |
| 35400 | 10000 | 10000 | 10000 | 9.013591 | 8.951243 | 9.511964 |
| 35700 | 10000 | 10000 | 10000 | 9.008431 | 8.946082 | 9.506803 |
| 36000 | 10000 | 10000 | 10000 | 8.989165 | 8.926816 | 9.487537 |
| 36300 | 10000 | 10000 | 10000 | 8.978297 | 8.915948 | 9.476669 |
| 36600 | 10000 | 10000 | 10000 | 8.968131 | 8.905782 | 9.466503 |
| 36900 | 10000 | 10000 | 10000 | 8.948976 | 8.886627 | 9.447348 |
| 37200 | 10000 | 10000 | 10000 | 8.944111 | 8.881762 | 9.442483 |
| 37500 | 10000 | 10000 | 10000 | 8.936353 | 8.874004 | 9.434725 |
| 37800 | 10000 | 10000 | 10000 | 8.921601 | 8.859252 | 9.419974 |
| 38100 | 10000 | 10000 | 10000 | 8.915529 | 8.853180 | 9.413901 |
| 38400 | 10000 | 10000 | 10000 | 8.908025 | 8.845676 | 9.406397 |
| 38700 | 10000 | 10000 | 10000 | 8.897130 | 8.834781 | 9.395502 |
| 39000 | 10000 | 10000 | 10000 | 8.893695 | 8.831347 | 9.392068 |
| 39300 | 10000 | 10000 | 10000 | 8.886855 | 8.824506 | 9.385227 |
| 39600 | 10000 | 10000 | 10000 | 8.877790 | 8.815442 | 9.376163 |
| 39900 | 10000 | 10000 | 10000 | 8.874872 | 8.812523 | 9.373244 |

This final holdout check was run once. No model, policy, or parameter was changed after reviewing these holdout results.
