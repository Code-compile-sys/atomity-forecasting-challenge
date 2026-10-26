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

## Workload shape

- `active_vm_count` is constant at 10,000 for every training timestamp. It provides no variation to forecast within this window; any forecast should respect the count’s nonnegative interpretation and should not claim evidence of churn from this segment.
- Aggregate CPU utilization averages 9.414%, with a standard deviation of 0.194 percentage points and a training range of 9.112%–9.912%.
- By contrast, the distribution of the 1.1 million VM-level readings is much wider: mean 9.414%, median 5.120%, 75th percentile 9.648%, 95th percentile 37.293%, and maximum 99.488%. This right-skew and spread show that the aggregate mean conceals substantial VM-level heterogeneity.
- Training-period CPU peaks occur around raw timestamps 16,500–20,400 seconds; the maximum aggregate CPU value is approximately 9.912%. The smallest value is approximately 9.112% at raw timestamp 32,400 seconds. These are local fluctuations, not evidence of recurring daily cycles.

## Segment differences

Training VM-level mean CPU differs across the available metadata segments:

| Segment | Training mean VM CPU |
|---|---:|
| Delay-insensitive | 8.257% |
| Interactive | 11.505% |
| Unknown category | 10.355% |

By virtual-core bucket, the observed means are 9.764% for 2 cores, 7.919% for 4, 9.658% for 8, 11.429% for 24, and 6.619% for `>24`. The larger buckets have far fewer VMs, so these differences are descriptive and should not be treated as stable population estimates. The notebook also reports the memory-bucket breakdown directly from training readings.

## Modeling implications and caveats

- With only about 9 hours of training history and five-minute sampling, the data cannot support claims about daily or weekly seasonality. Calendar seasonal features would be weakly justified for this source-relative trace.
- Since the active count is flat in training, a simple persistence forecast is an important baseline for that target. CPU utilization changes modestly at the aggregate level, while individual VM readings vary considerably.
- Segment-level CPU differences may motivate diagnostics, but using metadata CPU-summary fields would leak precomputed target-like information and is prohibited. Only segment labels and training-period readings were used here.
- EDA and all model/hyperparameter choices must remain inside the 110-timestamp training partition. Use rolling-origin validation there; reserve the final 24 timestamps for the required final evaluation only.
