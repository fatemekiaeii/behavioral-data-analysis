# Behavioral Data Analysis
# Behavioral Data Analysis — HCI Eye-Tracking Study

Python pipeline for analyzing multimodal behavioral data from a longitudinal 
eye-tracking study on automaticity disruption and recovery following interface change.

## Research Context

This pipeline was developed to test four hypotheses about how users adapt to 
interface updates:

- **H1**: Interface updates cause immediate disruption to automatic behavior
- **H2**: Users gradually recover automaticity following interface change
- **H3**: Higher pre-update automaticity predicts greater immediate disruption
- **H4**: Anticipatory guidance accelerates post-disruption recovery

## Behavioral Constructs

| Construct | Operationalization | Data Source |
|---|---|---|
| Attentional organization | Negative gaze-transition entropy | Eye-tracking |
| Behavioral organization | Negative behavioral-transition entropy | Interaction logs |
| Motor efficiency | Negative normalized AUC of mouse trajectories | Interaction logs |
| Task-execution efficiency | 1 − (unique actions / total actions) | Interaction logs |

## Statistical Methods

- **Interrupted time series (ITS)** with piecewise mixed-effects models for H1–H3
- **OLS with participant-clustered standard errors** for H4 early-window tests
- **BH-FDR correction** applied to all focal hypothesis tests
- Random slopes for time within participants

## Pipeline Structure
behavioral-data-analysis/
├── Automaticity_Analysis.py   # Main analysis script
├── requirements.txt           # Dependencies
└── outputs/                   # Generated CSV files
├── eye_panel_final.csv
├── log_panel_final.csv
├── integrated_its_H1_H3_summary.csv
├── early_window_H4_summary.csv
└── final_hypothesis_summary_combined.csv
## Input Data

The script expects two input files in the working directory:

| File | Format | Description |
|---|---|---|
| `binned_entropy_pupil_60sec_dataset.csv` | CSV | Binned eye-tracking data with gaze entropy measures |
| `combined_log_data.xlsx` | Excel | Raw interaction log data across all participants and phases |

## Usage

```bash
pip install -r requirements.txt
python Automaticity_Analysis.py
```

## Output Files

| File | Description |
|---|---|
| `eye_panel_final.csv` | Processed eye-tracking panel |
| `log_panel_final.csv` | Processed interaction-log panel |
| `integrated_its_H1_H3_summary.csv` | H1–H3 ITS model results |
| `early_window_H4_summary.csv` | H4 guidance buffering results |
| `final_hypothesis_summary_combined.csv` | Combined hypothesis summary with FDR-corrected p-values |

## Study Design

- **Participants**: 52 recruited through HEC Montréal participant panel
- **Design**: Mixed longitudinal (within: update phase; between: guidance condition)
- **Phases**: Baseline (10 min) → Update 1 (10 min) → Update 2 (10 min)
- **Conditions**: 6 counterbalanced update sequences (S1–S6)
- **Ethics**: Approved by HEC Montréal Research Ethics Committee

## Related Repository

Experimental platform: [tetris-hci-experiment](https://github.com/fatemekiaeii/tetris-hci-experiment)

## Citation

Kiaei Alamdari, F., Léger, P.-M., & Ortiz de Guinea, A. *Automaticity 
Disruption and Re-Habituation Following Interface Updates: A Longitudinal 
Eye-Tracking and Interaction-Log Study.* Working paper.
