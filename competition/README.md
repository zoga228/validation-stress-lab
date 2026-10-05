# Airline Satisfaction — competition application

| Experiment | OOF AUC | OOF log loss |
|---|---:|---:|
| baseline-leaves31 | 0.958662 | 0.227624 |
| crossfit-target-encoding | 0.960140 | 0.225494 |
| services-leaves31 | 0.958383 | 0.228165 |

Fixed 3-fold stratified CV (seed 42), no id feature. Early stopping uses outer validation, so CV is a model-selection estimate and can be optimistic; final private performance is unknown. The first service-feature comparison did not improve CV and was not submitted. Cross-fitted target encoding improved CV and qualified for a separate submission. Only CV improvement determines submission; public scores do not select configurations.

## Recorded public results

| Submitted experiment | Public ROC AUC |
|---|---:|
| crossfit-target-encoding | 0.95970 |
| baseline-leaves31 | 0.95821 |

Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition after accepting its rules; place them under `data/`. Install `requirements.txt` and run `python train.py --index 0` or `--index 1`. Later configurations are bounded experiments, not claimed results until measured. Competition data, raw row predictions and trained competition models are excluded from Git.

[Executable Kaggle notebook](https://www.kaggle.com/code/zangar09417/s6e10-fixed-fold-cv-and-feature-ablations). This Playground does not award competition ranking points.
