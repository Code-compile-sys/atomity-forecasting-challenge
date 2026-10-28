# Cloud Workload Forecasting and Capacity Planning

## 1. Executive summary

This project forecasts two measures from the Azure VM trace: `active_vm_count` and `aggregate_cpu_utilization`. Readings arrive every five minutes. The first 110 timestamps are used for model training and selection; the final 24 timestamps make up the two-hour holdout. The original timestamps, measured in seconds from trace start, are kept in the output. A separate UTC datetime is used to order and model the data; it is not the workload's actual local time.

We compared three approaches: the average of the last 12 readings (`Last12Mean`), Ridge regression using past CPU values, and damped exponential smoothing (`DampedETS`). We compared them using rolling-origin cross-validation (CV) on the first 110 timestamps only. For aggregate CPU, Ridge had the lowest average CV MAE: 0.224 percentage points, compared with 0.260 for Last12Mean and 0.244 for DampedETS. Ridge improved enough over the baseline to pass the 3% selection rule, so it was selected for CPU.

The final holdout check was run once after the model and capacity policy were selected. On the CPU holdout, Ridge had MAE 0.3012 and RMSE 0.3604; Last12Mean had MAE 0.1230 and RMSE 0.1613. The baseline did better on this one holdout. These scores are reported as a final check only; they were not used to change the model, policy, or settings.

For capacity planning, we recommend adding the full upper interval margin to the forecast when avoiding CPU shortfalls is the priority. On CV, this policy was below actual CPU in 10.12% of cases, compared with 82.14% for using the forecast alone. The trade-off is more unused capacity. The short data window and weaker CPU result on the final holdout mean these findings are useful as a starting point, not a guarantee of production performance.

## 2. Data understanding and preparation

The release contains VM-level CPU readings and VM metadata. To build the required series, readings are grouped by their raw timestamp. At each timestamp, `active_vm_count` is the number of distinct VMs with a valid CPU reading, and `aggregate_cpu_utilization` is the average of the VMs' `avg_cpu` values. Rows with a missing timestamp, VM ID, or CPU reading are excluded. The original timestamp is retained, and a separate UTC datetime is added for time-based ordering and modeling. Metadata CPU summaries (`max_cpu`, `avg_cpu`, and `p95_max_cpu`) are not used as model inputs.

All reported training analysis and model choices use the first 110 timestamps. The last 24 are held out. The training timestamps run from raw timestamp 0 through 32,700 seconds (source-relative UTC 00:00 through 09:05) and are spaced exactly 300 seconds apart. There are no gaps or duplicate aggregate timestamps. The training readings contain 1,100,000 VM-level rows. They have no missing timestamps, VM IDs, or `avg_cpu` values, no repeated `(timestamp, vm_id)` pairs, and no `avg_cpu` values outside the documented 0–100 range.

Within the 110-point training window, `active_vm_count` is constant at 10,000. This is a survivorship effect of the observed cohort: the selected VMs appear throughout the observed training interval, so this target describes the observed group and does not show whether a wider cloud fleet has arrivals or departures.

## 3. Workload analysis

### Overall CPU use and change over time

During training, aggregate CPU averages 9.414%, has a standard deviation of 0.194 percentage points, and ranges from 9.112% to 9.912%. Its trailing one-hour average falls from 9.411% at the first full window to 9.216% at the last training timestamp. This is a small decline in this short sample, not proof of a long-term trend. The highest single aggregate reading is 9.912% at raw timestamp 16,500 seconds. High readings occur around raw timestamps 16,500–20,400 seconds.

**Why it matters:** Average demand is fairly steady, but it does change. A capacity plan should account for possible increases rather than rely only on the overall average.

The training data covers about nine hours. Mean CPU by elapsed hour rises from 9.361% in hour 3 to 9.707% in hour 5, then falls. Hour 9 has a mean of 9.138%, but is based on just two readings. Only one source-relative weekday appears (all 110 timestamps map to Thursday UTC). The correlation between values one hour apart is 0.367; between values two hours apart it is -0.026. These results do not establish a repeating daily or weekly pattern.

**Why it matters:** The hourly changes may help with short-term monitoring, but the partial final hour and lack of repeated days are not enough to build a reliable clock-based schedule.

![Mean aggregate CPU by elapsed hour; hour 9 is based on two observations.](figures/hourly_cpu_means.png)

### Individual VMs and workload groups

Individual VM readings vary much more than the aggregate average. Their mean is 9.414%, median 5.120%, 75th percentile 9.648%, 95th percentile 37.293%, and maximum 99.488%. A total of 4,635 VMs (46.35%) average below 5% CPU; 220 (2.20%) average above 50%.

**Why it matters:** A stable fleet average does not mean every VM is lightly loaded. Individual VMs may be much busier, so fleet-wide averages should be considered alongside machine-level monitoring.

The supplied metadata groups show different CPU levels and group sizes. Interactive VMs average 11.505% CPU and contribute 37.776% of summed CPU readings. Delay-insensitive VMs average 8.257% and contribute 54.206%; Unknown VMs average 10.355% and contribute 8.018%.

| Workload category | VM count | Average CPU | Share of summed CPU readings |
|---|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.257% | 54.206% |
| Interactive | 3,091 | 11.505% | 37.776% |
| Unknown | 729 | 10.355% | 8.018% |

The 2-core group contributes 78.142% of summed CPU readings. The 24-core group has the highest average CPU, 11.429%, but contains only 62 VMs.

| Virtual-core group | VM count | Average CPU | Share of summed CPU readings |
|---|---:|---:|---:|
| 2 | 7,534 | 9.764% | 78.142% |
| 4 | 1,908 | 7.919% | 16.049% |
| 8 | 486 | 9.658% | 4.986% |
| 24 | 62 | 11.429% | 0.753% |
| >24 | 10 | 6.619% | 0.070% |

The 2-GB memory group contributes 36.503% of summed CPU readings; the 4-GB group has the highest average among the larger groups, at 11.460%.

| Memory group (GB) | VM count | Average CPU | Share of summed CPU readings |
|---|---:|---:|---:|
| 2 | 3,904 | 8.802% | 36.503% |
| 4 | 2,475 | 11.460% | 30.128% |
| 8 | 2,146 | 9.143% | 20.843% |
| 32 | 1,383 | 7.814% | 11.479% |
| 64 | 82 | 11.217% | 0.977% |
| >64 | 10 | 6.619% | 0.070% |

**Why it matters:** Capacity planning should consider both how busy a group is and how many VMs it contains. The 24-core group has a high average but a small total share. The shares above refer to summed CPU readings, not hardware capacity. Results for small groups, especially 24-core, >24-core, 64-GB, and >64-GB, may be less reliable because those groups have few VMs. These summaries describe the data; they do not prove that group membership causes different CPU use.

## 4. Forecasting approach

We compared exactly three candidates:

1. **Last12Mean:** averages the most recent 12 values, or one hour of readings. This is the simple baseline.
2. **Ridge:** learns a relationship between a value and recent history. It uses lags 1, 2, and 6, plus trailing means and standard deviations over 3, 6, and 12 readings. To forecast multiple steps, it uses each forecast as input to the next step.
3. **Damped ETS:** exponential smoothing with a damped trend, fit with statsmodels. It has no seasonal component because this dataset is too short to establish a repeating season.

All models use the target's past values only. Future target values are not provided to the forecasters. Count forecasts are kept at zero or above; CPU forecasts are kept between 0 and 100. No CPU-summary metadata is used as a model input.

The model is chosen separately for each target using average CV MAE. Ridge or DampedETS must improve on Last12Mean by at least 3%; otherwise, Last12Mean is selected. If Ridge and DampedETS tie, DampedETS is preferred as the simpler model.

## 5. Validation method

We used expanding rolling-origin cross-validation on the training data. Each fold starts with at least 50 observations, forecasts the next 24, and moves forward by 6 observations to create the next fold. This produces seven folds. The validation windows overlap, so the 168 forecast cases are not 168 separate timestamps.

This approach tests the models in time order. At each step, the model uses only data that would have been available at that point, then forecasts the next two hours. Unlike a random split, it does not mix future points into the training data. The final 24 timestamps are not used for cross-validation, model selection, or fitting.

We report MAE and RMSE for each fold and average them across folds. CPU errors are in percentage points. The existing focused comparison does not report numeric values for normalized MAE or sMAPE, so this report does not add them.

## 6. Results compared with the baseline

The table shows average CV errors on the training data. Lower errors are better.

| Target | Model | Mean CV MAE | Mean CV RMSE |
|---|---|---:|---:|
| `active_vm_count` | Last12Mean | 0.000 | 0.000 |
| `active_vm_count` | Ridge | 0.000 | 0.000 |
| `active_vm_count` | DampedETS | 0.000 | 0.000 |
| `aggregate_cpu_utilization` | Last12Mean | 0.260 | 0.288 |
| `aggregate_cpu_utilization` | Ridge | 0.224 | 0.255 |
| `aggregate_cpu_utilization` | DampedETS | 0.244 | 0.275 |

All models have zero error for VM count because it does not change in training. For CPU, Ridge has the lowest average CV MAE and RMSE. It also passes the 3% improvement rule, so the selection rule chooses Ridge.

![Mean CPU CV MAE and RMSE for the three candidates.](figures/cv_cpu_scores.png)

### Final holdout check (run once)

After model selection, the selected models were fit on the first 110 points and used to forecast the final 24. Holdout actuals were used only to calculate these scores. No model, policy, or parameter was changed after seeing them.

| Target | Selected model | Selected MAE | Selected RMSE | Baseline | Baseline MAE | Baseline RMSE |
|---|---|---:|---:|---|---:|---:|
| `active_vm_count` | Last12Mean | 0.0000 | 0.0000 | Last12Mean | 0.0000 | 0.0000 |
| `aggregate_cpu_utilization` | Ridge | 0.3012 | 0.3604 | Last12Mean | 0.1230 | 0.1613 |

On this holdout, Last12Mean did better than Ridge for CPU. This differs from the CV result and is an important limitation; it is not a reason to change the model after seeing the holdout. The full forecast and interval table for all 24 raw timestamps is in `notes.md`.

## 7. Forecast errors and where they occur

“Peak” means actual CPU above the 90th percentile of the training series: 9.6684%. The seven overlapping folds contain 168 cases: 27 peak cases and 141 non-peak cases.

| Group | Cases | MAE | RMSE | Average error (actual − forecast) |
|---|---:|---:|---:|---:|
| Peak (`actual > 9.6684%`) | 27 | 0.4104 | 0.4611 | +0.4104 |
| Non-peak | 141 | 0.1885 | 0.2279 | +0.1400 |

Peak errors are more than twice as large as non-peak errors. The positive average error in both groups means Ridge tends to forecast too low, especially during peaks.

The folds with the highest MAE are:

| Rank | Fold | MAE |
|---:|---:|---:|
| 1 | 1 | 0.4148 |
| 2 | 2 | 0.3296 |
| 3 | 3 | 0.2523 |
| 4 | 6 | 0.1655 |
| 5 | 5 | 0.1545 |
| 6 | 7 | 0.1446 |
| 7 | 4 | 0.1076 |

In Fold 1, actual CPU rises to a sustained high near 9.91%, while Ridge stays around 9.2–9.4%. This suggests the forecast did not rise quickly enough. In Fold 2, Ridge gradually predicts lower values while actual CPU remains high, then actual CPU drops near the end. The forecast misses both changes. These are likely explanations, not proven causes.

**Why it matters:** Errors are concentrated in high CPU periods and in early folds. If used for planning, the forecast needs a safety margin and monitoring for changes in demand. The evidence does not justify adding a separate peak model.

The optional error-by-step analysis found the lowest MAE was **0.1092 at step 2** and the highest was **0.3798 at step 16**. Error does not rise steadily with each step, so this short series does not support a simple claim that later steps always have larger errors.

## 8. Capacity recommendation

The three capacity policies use CPU CV forecasts from Ridge. The extra margin is the 90th percentile of the pooled CV errors (actual minus forecast): **0.4984 percentage points**.

- **Policy A:** provide exactly the forecast.
- **Policy B:** provide the forecast plus half the margin.
- **Policy C:** provide the forecast plus the full margin.

“Under-provisioning” means actual CPU is higher than the capacity provided. Its rate is the share of all cases where that happens. Average shortfall is calculated only for those cases. “Wasted headroom” is the extra capacity on cases where the provision is enough.

| Policy | Under-provisioning rate | Average shortfall | Wasted headroom |
|---|---:|---:|---:|
| A: forecast | 82.14% | 0.2481 | 0.1139 |
| B: forecast + half margin | 32.74% | 0.1679 | 0.1795 |
| C: forecast + full margin | 10.12% | 0.0868 | 0.3601 |

![CV shortfall and headroom trade-off for the three provisioning policies.](figures/capacity_policy_tradeoff.png)

**Recommendation:** Choose Policy C when avoiding shortfalls matters most. It has the lowest CV under-provisioning rate (10.12%) and average shortfall (0.0868), but the most wasted headroom (0.3601 percentage points). Policy B is a middle option if spare capacity is expensive.

On the one-time holdout, Policy C was below actual CPU at 6 of 24 points (25.00%). For the 18 points where it provided enough, average wasted headroom was 0.3151 percentage points. This holdout result is reported for transparency and did not change the policy choice.

## 9. Limitations and future work

The training data covers only about nine hours. It cannot show daily or weekly patterns, and the timestamps do not identify the real local time. The CV folds overlap, the data contains one observed VM group, and some metadata groups have few VMs. Ridge has the best CPU CV score, but Last12Mean did better on the one-time holdout. These results describe this trace and should not be treated as a general performance guarantee. Numeric normalized-MAE and sMAPE results are not available in the documented three-model comparison, so they are not reported here.

The `active_vm_count` remains at 10,000 for the full training window. This is a survivorship effect in the observed group: these VMs are present throughout the sample. It does not show how a changing fleet would behave, so conclusions about fleet growth or churn require data that includes such changes.

Future work should use longer traces and more forecast periods, check whether this VM group represents the fleet where forecasts would be used, and reassess the safety margin with more data. Any further model selection should continue to use training-only time-ordered validation and must not use the final holdout to tune the model.

---

All numbers in this report come from `notes.md`. The figures show those same documented values; no new measurements or holdout analysis were added.
