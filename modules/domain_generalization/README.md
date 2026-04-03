# Domain Generalization Module

Runs leave-one-domain-out evaluation to estimate out-of-domain robustness.

## What it does

- Repeats train/test by leaving one domain out each fold.
- Reports PR-AUC and ROC-AUC by held-out domain.

## Primary artifact

- lodo_results.csv

## Config section

- modules.domain_generalization
