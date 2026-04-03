# Evidence Bundle

This folder contains only final-round evidence used by the app, notebook, and pitch.

## Files

1. evidence_latest_run.json
   - Points to the best validated run and stores metrics/hash evidence.

2. final_benchmark_latest.json
   - Points to the final strict benchmark package.

3. benchmark_summary.json
   - Main benchmark evidence file.
   - Scope: strict latest ensemble vs TimeSFM proxy.

4. benchmark_full_sample_metrics.csv
5. benchmark_subsample_summary.csv
6. benchmark_winner_by_fraction.csv
   - Supporting benchmark tables for reviewer verification.

7. benchmark_README.md
   - Original benchmark package documentation.

## Key verified benchmark snapshot

1. Alignment rows: 59756 (labels_match = true)
2. Evaluated rows after stratified downsample: 10000
3. Ensemble PR-AUC: 0.7145
4. TimeSFM proxy PR-AUC: 0.0794
5. Delta PR-AUC (ensemble - TimeSFM proxy): 0.6351

Use benchmark_summary.json as source of truth for final spoken claims.
