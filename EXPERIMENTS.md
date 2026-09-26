# Experiment Log

Scored with `src/metrics.py` (macro F_0.5). Exps 0-2 run on the full 20k dev
sample (`sample/`, seed 42) — honest because these baselines have no learned
parameters or tuned thresholds. From the first learned model on, all scores
come from a held-out validation split (see VALIDATION section in README).

| Exp | Change | F0.5 | Prec | Rec | Singleton acc | Conclusion |
|-----|--------|------|------|-----|---------------|------------|
| 0 | Predict empty for all | 0.0544 | 0.0544 | 0.2932->0.0544* | 1.0000 | Floor = singleton rate (1089/20000). Pipeline wiring works. |
| 1 | Normalized exact-name match within country | 0.4606 | 0.5975 | 0.2932 | 0.9146 | Strong start: verbatim copies carry recall to 29%. Errors: generic-name FPs + variant/script FNs (see report). |

*Exp0 P/R equal the singleton rate by construction (1.0 on singletons, 0.0 elsewhere).
