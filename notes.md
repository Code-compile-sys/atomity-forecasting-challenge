# Training-only EDA notes

## What data was used

The target series was built from VM-level readings, then split by time: the first 110 timestamps are training data and the final 24 are held out. All EDA numbers and model choices below use only the 110 training timestamps and their VM readings. The holdout values were not used for those calculations or choices.

The original timestamps are seconds from the start of the source trace. A UTC datetime column is used to order and model the data, but it does not represent the workload's actual local date or time. Metadata CPU summary columns (`max_cpu`, `avg_cpu`, and `p95_max_cpu`) were not used as features.

## Data coverage and quality

- The training data has 110 timestamps at five-minute intervals, from raw timestamp 0 to 32,700 seconds (source-relative UTC 00:00 to 09:05).
- The interval between timestamps is always 300 seconds. There are no gaps or duplicate aggregate timestamps.
- The training readings contain 1,100,000 rows and 10,000 distinct VMs.
- No training rows have a missing timestamp, VM ID, or `avg_cpu`. There are no duplicate `(timestamp, vm_id)` pairs.
- All training `avg_cpu` values are between 0 and 100 percent.
- These timestamps are seconds from trace start, not real-world clock times. Do not interpret the UTC conversion as the actual time of day.

### Timestamp checks (training only)

| Check | Result | What it means |
|---|---:|---|
| Time between readings | 300 seconds | The series is sampled every 5 minutes. |
| Missing time points | 0 | No training timestamps are missing. |
| Duplicate target timestamps | 0 | Each aggregated time point appears once. Multiple VM rows at one time are expected. |

## VM lifetimes and changes in the VM set (training only)

The observed lifetime here is the time between a VM's first and last reading in the training data. It may not be the VM's full lifetime: the data starts and ends at boundaries, so VMs may have existed before the first timestamp or continued after the last.

| Measure | Result | What it means |
|---|---:|---|
| Median observed VM span | 32,700 seconds (9.08 hours) | The typical VM appears throughout the observed training period. |
| Mean observed VM span | 32,700 seconds (9.08 hours) | All 10,000 VMs have the same observed first-to-last span. |
| VMs first seen after the start | 0 | No new VMs appear inside the observed training window. |
| VMs last seen before the end | 0 | No VMs disappear inside the observed training window. |

All 10,000 VMs appear at the first training timestamp and remain through the last one. Because the window edges hide what happened before and after the trace, this does not prove that the VMs were created or deleted at those times. We see no changes in the VM set within this short window, but cannot draw conclusions about the wider fleet.

## CPU use across individual VMs (training only)

Each number below is a VM's average CPU use across its training readings.

| Measure | Result | What it means |
|---|---:|---|
| VMs averaging below 5% CPU | 4,635 (46.35%) | Nearly half of the VMs have low average CPU use. |
| VMs averaging above 50% CPU | 220 (2.20%) | Few VMs have high average use; this does not rule out short spikes. |

## Aggregate CPU over time (training only)

The aggregate series is the mean of VM-level `avg_cpu` at each timestamp. Its trailing 12-reading average uses the current reading and the previous 11 readings (one hour total).

- The trailing one-hour average falls from 9.411% at its first complete window to 9.216% at the end of training. This is a small decline in this short sample, not proof of a long-term trend.
- Aggregate CPU has a standard deviation of 0.194 percentage points, and ranges from 9.112% to 9.912%.
- The highest aggregate CPU reading is 9.912% at raw timestamp 16,500 seconds. This is a local high point, not an anomaly identified by a formal rule.
- About nine hours of data is not enough to establish a daily or weekly pattern.

### Mean CPU by elapsed hour

Hours are counted from the start of the trace. Hour 9 is only partially observed.

| Elapsed hour | Mean aggregate CPU | Readings |
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

The highest full-hour mean is 9.707% in elapsed hour 5. The lowest is 9.138% in hour 9, but that value uses only two readings. CPU is higher around hours 4–5 and lower later in this sample. Treat this as a short-term observation, not a schedule for future demand.

Only one source-relative weekday appears (Thursday UTC). The autocorrelation is 0.367 at a one-hour lag and -0.026 at a two-hour lag. These values do not prove a repeating cycle; the training history is too short to establish one.

## Overall workload shape

- `active_vm_count` is 10,000 at every training timestamp. There is no change in count to learn from this sample. Forecasts should not be taken as evidence that a larger production fleet has no churn.
- Aggregate CPU averages 9.414%, with a standard deviation of 0.194 percentage points and a range of 9.112%–9.912%.
- Individual VM readings vary much more than the aggregate: mean 9.414%, median 5.120%, 75th percentile 9.648%, 95th percentile 37.293%, and maximum 99.488%.
- Aggregate CPU is highest around raw timestamps 16,500–20,400 seconds. Its lowest value is about 9.112% at raw timestamp 32,400 seconds. These are local changes, not evidence of a repeating daily cycle.

The aggregate average is steady even though individual VM CPU values vary widely. For operations, this means the overall average alone may hide busy individual VMs.

## Differences between VM groups

The group labels below are supplied in the metadata; no new group cutoffs were created:

- Workload category: `Delay-insensitive`, `Interactive`, `Unknown`.
- Virtual-core groups: `2`, `4`, `8`, `24`, `>24`.
- Memory groups in GB: `2`, `4`, `8`, `32`, `64`, `>64`.

The `>24` and `>64` labels are open-ended categories, not exact numbers. Metadata CPU summary fields were excluded.

For each group, average CPU is the mean of its VM-level `avg_cpu` readings during training. “Share of summed CPU readings” means 100 times the group's sum of CPU readings divided by the sum across all training VMs. It is not a share of hardware capacity because exact VM capacity is not available. Reading coverage is balanced across VMs in this training slice.

### Workload category

| Category | VM count | Average CPU | Share of summed CPU readings |
|---|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.257% | 54.206% |
| Interactive | 3,091 | 11.505% | 37.776% |
| Unknown | 729 | 10.355% | 8.018% |

Interactive VMs have the highest average CPU among the large categories, while Delay-insensitive VMs contribute the largest total share because there are more of them. Capacity planning should consider both how busy a group is and how many VMs it contains.

### Virtual-core group

| Core group | VM count | Average CPU | Share of summed CPU readings |
|---|---:|---:|---:|
| 2 | 7,534 | 9.764% | 78.142% |
| 4 | 1,908 | 7.919% | 16.049% |
| 8 | 486 | 9.658% | 4.986% |
| 24 | 62 | 11.429% | 0.753% |
| >24 | 10 | 6.619% | 0.070% |

The 2-core group accounts for most summed CPU readings (78.142%). The 24-core group has the highest average (11.429%), but contains only 62 VMs. A high average in a small group does not mean that group drives most total demand.

### Memory group

| Memory group (GB) | VM count | Average CPU | Share of summed CPU readings |
|---|---:|---:|---:|
| 2 | 3,904 | 8.802% | 36.503% |
| 4 | 2,475 | 11.460% | 30.128% |
| 8 | 2,146 | 9.143% | 20.843% |
| 32 | 1,383 | 7.814% | 11.479% |
| 64 | 82 | 11.217% | 0.977% |
| >64 | 10 | 6.619% | 0.070% |

The 2-GB group contributes the largest share (36.503%). The 4-GB group has the highest average CPU among the larger groups (11.460%). Group averages alone do not tell us how much total capacity each group needs.

These are descriptions of this training sample, not proof that group membership causes different CPU use. Results for small groups, especially 24-core, >24-core, 64-GB, and >64-GB, may be less reliable because they contain few VMs.

## What the data means for forecasting

- The training history is about nine hours long, so it cannot support daily or weekly seasonal forecasts.
- The VM count is flat, making a simple average forecast a useful baseline for that target.
- Aggregate CPU changes modestly, while individual VM readings vary much more.
- Group labels can help describe workloads, but metadata CPU summaries (`max_cpu`, `avg_cpu`, `p95_max_cpu`) must not be used as features because they contain CPU information.
- Model choices use rolling-origin validation within the first 110 timestamps. The final 24 timestamps are kept for the one-time final evaluation.

## Comparing the three forecasting candidates (training-only CV)

The evaluator compares exactly three candidates for each target: the average of the last 12 values (`Last12Mean`), Ridge, and damped exponential smoothing (`DampedETS`). For each, it reports MAE and RMSE for each fold and the average across folds. The metric helpers also guard normalized MAE and sMAPE against division by zero, but those metrics are not part of this focused comparison.

| Target | Model | Mean MAE | Mean RMSE |
|---|---|---:|---:|
| `active_vm_count` | Last12Mean | 0.000 | 0.000 |
| `active_vm_count` | Ridge | 0.000 | 0.000 |
| `active_vm_count` | DampedETS | 0.000 | 0.000 |
| `aggregate_cpu_utilization` | Last12Mean | 0.260 | 0.288 |
| `aggregate_cpu_utilization` | Ridge | 0.224 | 0.255 |
| `aggregate_cpu_utilization` | DampedETS | 0.244 | 0.275 |

There are seven folds per target. Each fold starts with at least 50 training points, forecasts 24 points, then advances the starting point by 6. All candidates have zero error on the constant VM-count series. Ridge has the lowest average MAE and RMSE for CPU in CV. No holdout target values are used here.

### Fold boundaries

Rows are ordered by `datetime`. Row ranges use `[start:end)`, meaning the start row is included and the end row is not.

| Fold | Training rows | Validation rows |
|---:|---|---|
| 1 | `[0:50)` | `[50:74)` |
| 2 | `[0:56)` | `[56:80)` |
| 3 | `[0:62)` | `[62:86)` |
| 4 | `[0:68)` | `[68:92)` |
| 5 | `[0:74)` | `[74:98)` |
| 6 | `[0:80)` | `[80:104)` |
| 7 | `[0:86)` | `[86:110)` |

## Optional 80% prediction intervals

For each target, the model is selected using CV on training data. Its errors from all CV folds and forecast steps are pooled. An error is actual minus forecast. The 10th and 90th percentiles of these errors are added to the final forecast to form the 80% interval. Count forecasts are clipped to zero or above; CPU forecasts and bounds are clipped to `[0,100]`. The interval output is separate from the required point-prediction CSV.

## Aggregate CPU error analysis (training-only CV)

Ridge is selected for CPU. “Peak” means an actual training-series value above its 90th percentile, 9.6684%. The seven folds contain 168 forecast cases. Folds overlap, so some timestamps appear in more than one case.

| Group | Forecast cases | MAE | RMSE | Average error (actual − forecast) |
|---|---:|---:|---:|---:|
| Peak (`actual > 9.6684%`) | 27 | 0.4104 | 0.4611 | +0.4104 |
| Non-peak | 141 | 0.1885 | 0.2279 | +0.1400 |

Errors are larger during peaks—more than twice the non-peak MAE. The positive average error in both groups means Ridge tends to forecast too low, especially at peaks.

### Folds ranked by MAE

| Rank | Fold | MAE |
|---:|---:|---:|
| 1 | 1 | 0.4148 |
| 2 | 2 | 0.3296 |
| 3 | 3 | 0.2523 |
| 4 | 6 | 0.1655 |
| 5 | 5 | 0.1545 |
| 6 | 7 | 0.1446 |
| 7 | 4 | 0.1076 |

- **Fold 1 likely reason:** Actual CPU rises to a sustained high point (9.91%), but Ridge stays around 9.2–9.4% and does not rise enough.
- **Fold 2 likely reason:** Ridge gradually forecasts lower values while actual CPU stays high, then actual CPU drops near the end. The forecast misses both changes.

The largest errors occur in early validation folds, where the recent training history is less like the values that follow. These are likely explanations, not proven causes.

### Error by forecast step

Across folds, the lowest step MAE is **0.1092 at step 2**; the highest is **0.3798 at step 16**. Error does not increase steadily with each step, so this short series does not support a simple rule that forecasts always get worse at longer horizons.

## Comparing CPU capacity policies (training-only CV)

The selected CPU model is Ridge. The extra margin is the 90th percentile of its pooled CV errors: **0.4984 CPU percentage points**. The table uses all 168 CV cases, not the final holdout.

| Policy | Under-provisioned cases | Average shortfall | Wasted headroom |
|---|---:|---:|---:|
| A: forecast only | 82.14% | 0.2481 | 0.1139 |
| B: forecast + half the upper interval margin | 32.74% | 0.1679 | 0.1795 |
| C: forecast + full upper interval margin | 10.12% | 0.0868 | 0.3601 |

“Under-provisioned” means actual CPU is higher than the amount provided. Its rate is the share of all cases where that happens. Average shortfall is actual minus provision, averaged only over those cases. “Wasted headroom” is provision minus actual, averaged only over cases with enough capacity. The last two columns are CPU percentage points.

**Recommendation:** Use Policy C if avoiding CPU shortfalls is the main priority. It has the lowest CV shortfall rate (10.12%) and smallest average shortfall (0.0868), but leaves the most unused headroom (0.3601 percentage points). Policy B is a compromise if spare capacity is costly.

## Final holdout check (run once)

The chosen models were fit on the first 110 points and used to forecast the final 24 timestamps. Their CV errors were used for interval bounds. Actual holdout values were used only to calculate the scores below—not to choose or change a model, policy, or parameter.

| Target | Selected model | Selected MAE | Selected RMSE | Baseline | Baseline MAE | Baseline RMSE |
|---|---|---:|---:|---|---:|---:|
| `active_vm_count` | Last12Mean | 0.0000 | 0.0000 | Last12Mean | 0.0000 | 0.0000 |
| `aggregate_cpu_utilization` | Ridge | 0.3012 | 0.3604 | Last12Mean | 0.1230 | 0.1613 |

On the CPU holdout, Policy C was below actual CPU at 6 of 24 points (25.00%). On the other 18 points, average wasted headroom was 0.3151 percentage points.

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

This final holdout check was run once. No model, policy, or parameter was changed after seeing its results.
