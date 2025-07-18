# HRV Metrics Tables - Example Output

This document shows what the HRV metrics tables would look like when you run the code on your Valquiria dataset.

## Basic Summary Table

| Subject      | Sol  | n_beats | mean_hr_bpm | std_hr_bpm | mean_rr_ms | std_rr_ms | cvnn  |
|-------------|------|---------|-------------|------------|------------|-----------|-------|
| T01_Mara    | Sol2 | 1200    | 68.5        | 8.2        | 876.8      | 105.2     | 12.0  |
| T01_Mara    | Sol3 | 1150    | 72.1        | 9.1        | 833.5      | 112.8     | 13.5  |
| T02_Laura   | Sol2 | 1300    | 65.2        | 7.5        | 920.2      | 98.7      | 10.7  |
| T02_Laura   | Sol3 | 1250    | 69.8        | 8.8        | 859.1      | 108.9     | 12.7  |
| T03_Nancy   | Sol2 | 1100    | 71.3        | 9.5        | 842.1      | 115.4     | 13.7  |

## Time Domain HRV Metrics

| Subject      | Sol  | mean_nni | sdnn  | rmssd | nn50 | pnn50 | nn20 | pnn20 | cvnn  |
|-------------|------|----------|-------|-------|------|-------|------|-------|-------|
| T01_Mara    | Sol2 | 876.8    | 98.4  | 45.2  | 154  | 12.8  | 420  | 35.0  | 11.2  |
| T01_Mara    | Sol3 | 833.5    | 105.7 | 52.1  | 213  | 18.5  | 465  | 40.4  | 12.7  |
| T02_Laura   | Sol2 | 920.2    | 89.2  | 38.9  | 116  | 8.9   | 380  | 29.2  | 9.7   |
| T02_Laura   | Sol3 | 859.1    | 102.3 | 47.8  | 190  | 15.2  | 445  | 35.6  | 11.9  |
| T03_Nancy   | Sol2 | 842.1    | 108.9 | 51.2  | 210  | 19.1  | 470  | 42.7  | 12.9  |

## Frequency Domain HRV Metrics

| Subject      | Sol  | total_power | vlf   | lf   | hf   | lf_hf_ratio | lfnu  | hfnu  |
|-------------|------|-------------|-------|------|------|-------------|-------|-------|
| T01_Mara    | Sol2 | 2140        | 850   | 1250 | 890  | 1.40        | 58.4  | 41.6  |
| T01_Mara    | Sol3 | 2700        | 1020  | 1580 | 1120 | 1.41        | 58.5  | 41.5  |
| T02_Laura   | Sol2 | 1700        | 720   | 980  | 720  | 1.36        | 57.6  | 42.4  |
| T02_Laura   | Sol3 | 2470        | 950   | 1420 | 1050 | 1.35        | 57.5  | 42.5  |
| T03_Nancy   | Sol2 | 2830        | 1050  | 1650 | 1180 | 1.40        | 58.3  | 41.7  |

## Nonlinear HRV Metrics

| Subject      | Sol  | sd1  | sd2   | sd1_sd2_ratio | ellipse_area |
|-------------|------|------|-------|---------------|--------------|
| T01_Mara    | Sol2 | 32.1 | 128.7 | 0.25          | 12985        |
| T01_Mara    | Sol3 | 36.9 | 138.4 | 0.27          | 16064        |
| T02_Laura   | Sol2 | 27.5 | 116.8 | 0.24          | 10095        |
| T02_Laura   | Sol3 | 33.8 | 134.1 | 0.25          | 14252        |
| T03_Nancy   | Sol2 | 36.2 | 142.5 | 0.25          | 16218        |

## HRV Metrics Explanation

### Basic Summary Metrics
- **n_beats**: Number of heartbeats in the analysis segment
- **mean_hr_bpm**: Average heart rate (beats per minute)
- **std_hr_bpm**: Standard deviation of heart rate
- **mean_rr_ms**: Average RR interval (milliseconds between heartbeats)
- **std_rr_ms**: Standard deviation of RR intervals
- **cvnn**: Coefficient of variation of RR intervals (normalized measure of variability)

### Time Domain Metrics
- **mean_nni**: Mean of RR intervals (same as mean_rr_ms)
- **sdnn**: Standard deviation of all RR intervals (measure of overall HRV)
- **rmssd**: Root mean square of successive differences (measure of short-term variability)
- **nn50**: Number of successive RR intervals differing by > 50ms
- **pnn50**: Percentage of successive RR intervals differing by > 50ms
- **nn20**: Number of successive RR intervals differing by > 20ms
- **pnn20**: Percentage of successive RR intervals differing by > 20ms

### Frequency Domain Metrics
- **total_power**: Total power in all frequency bands (ms²)
- **vlf**: Very low frequency power (0.003-0.04 Hz)
- **lf**: Low frequency power (0.04-0.15 Hz) - reflects sympathetic and parasympathetic activity
- **hf**: High frequency power (0.15-0.4 Hz) - reflects parasympathetic activity
- **lf_hf_ratio**: Ratio of LF to HF power - indicates autonomic balance
- **lfnu**: LF power in normalized units
- **hfnu**: HF power in normalized units

### Nonlinear Metrics
- **sd1**: Standard deviation perpendicular to line of identity in Poincaré plot (short-term variability)
- **sd2**: Standard deviation along line of identity in Poincaré plot (long-term variability)
- **sd1_sd2_ratio**: Ratio of SD1 to SD2
- **ellipse_area**: Area of the Poincaré plot ellipse

## Summary Statistics Example

### Overall Statistics
| Metric      | count | mean  | std   | min   | 25%   | 50%   | 75%   | max   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|
| mean_hr_bpm | 32    | 69.2  | 8.5   | 55.2  | 63.1  | 68.9  | 74.8  | 85.6  |
| sdnn        | 32    | 98.7  | 15.2  | 68.4  | 87.2  | 96.5  | 108.9 | 132.1 |
| rmssd       | 32    | 45.8  | 12.3  | 25.1  | 37.2  | 44.9  | 52.6  | 71.2  |
| lf_hf_ratio | 32    | 1.38  | 0.25  | 0.89  | 1.21  | 1.37  | 1.54  | 1.89  |

### Mean Values by Subject
| Subject      | mean_hr_bpm | sdnn  | rmssd | lf_hf_ratio |
|-------------|-------------|-------|-------|-------------|
| T01_Mara    | 70.3        | 102.1 | 48.7  | 1.41        |
| T02_Laura   | 67.5        | 95.8  | 43.4  | 1.36        |
| T03_Nancy   | 71.3        | 108.9 | 51.2  | 1.40        |
| T04_Michelle| 69.8        | 96.2  | 44.1  | 1.35        |

### Mean Values by Sol
| Sol  | mean_hr_bpm | sdnn  | rmssd | lf_hf_ratio |
|------|-------------|-------|-------|-------------|
| Sol2 | 68.1        | 98.7  | 44.3  | 1.38        |
| Sol3 | 70.4        | 102.3 | 49.7  | 1.40        |
| Sol4 | 69.2        | 96.5  | 43.8  | 1.35        |
| Sol5 | 68.9        | 100.1 | 46.2  | 1.39        |

## How to Use This Information

1. **Autonomic Balance**: Look at the `lf_hf_ratio` - values around 1.0-1.5 are typical
2. **Overall Variability**: Higher `sdnn` values indicate better overall HRV
3. **Short-term Variability**: Higher `rmssd` values indicate better parasympathetic activity
4. **Individual Differences**: Compare subjects to see who has better HRV profiles
5. **Temporal Changes**: Compare across Sols to see how HRV changes over time

## Clinical Significance

- **Higher HRV** (higher sdnn, rmssd) generally indicates better cardiovascular health
- **Lower LF/HF ratio** may indicate better parasympathetic activity
- **Individual baselines** are important - look at relative changes rather than absolute values
- **Stress response** can be seen in decreased HRV and increased LF/HF ratio

## Files Generated

When you run the code, you'll get these CSV files:
- `hrv_basic_summary.csv` - Basic heart rate statistics
- `hrv_time_domain.csv` - Time domain HRV metrics
- `hrv_frequency_domain.csv` - Frequency domain HRV metrics
- `hrv_nonlinear.csv` - Nonlinear HRV metrics
- `hrv_complete.csv` - All metrics combined
- `hrv_summary_statistics.csv` - Statistical summaries
- `hrv_by_subject.csv` - Metrics averaged by subject
- `hrv_by_sol.csv` - Metrics averaged by Sol 