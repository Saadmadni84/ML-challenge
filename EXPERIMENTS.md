# Experiment Log

Scored with `src/metrics.py` (macro F_0.5). Exps 0-2 run on the full 20k dev
sample (`sample/`, seed 42) — honest because these baselines have no learned
parameters or tuned thresholds. From the first learned model on, all scores
come from a held-out validation split (see VALIDATION section in README).

| Exp | Change | F0.5 | Prec | Rec | Singleton acc | Conclusion |
|-----|--------|------|------|-----|---------------|------------|
| 0 | Predict empty for all | 0.0544 | 0.0544 | 0.2932->0.0544* | 1.0000 | Floor = singleton rate (1089/20000). Pipeline wiring works. |
| 1 | Normalized exact-name match within country | 0.4606 | 0.5975 | 0.2932 | 0.9146 | Strong start: verbatim copies carry recall to 29%. Errors: generic-name FPs + variant/script FNs (see report). |
| 2a | TF-IDF char-ngram blocking, name-only (K=100) | n/a (blocking) | n/a | **0.860** ceiling (@25: 0.812; latin-only @100: 0.922; India 0.750 / US 0.933) | n/a | 97% of Latin misses share an address token → address channel justified. |
| 2b | TF-IDF char-ngram blocking, address-only (K=100) | n/a (blocking) | n/a | **0.941** ceiling (@25: 0.923) | n/a | Address alone beats name alone. Channels are complementary. |
| 2c | UNION name@50 + addr@50 (<=100/S1, avg 97.2) | n/a (blocking) | n/a | **0.9911** ceiling (latin 0.9944; India 0.9838 / US 0.9960; 97.1% S1s full recall) | n/a | Blocking SOLVED on sample. Residual 0.89%: 41% non-Latin, 37% empty-address, 0.2% hard noise. |
| 3 | LightGBM on 18 string feats (union cands), tau tuned on val | **0.9723** | 0.9825 | 0.9521 | 0.9679 | tau=0.96 (4k val S1s). Ref on same val: Exp0 0.0545, Exp1 0.4508. India 0.9583 / US 0.9816. FN:FP = 3:1; rejected truths are empty-addr / non-Latin / DBA cases. Top feats: addr_tridice, addr_tset, name_tsort. |
| 4 | +8 blocking-signal feats (v2, 26 total), retrain + re-tune | **0.9762** | 0.9833 | 0.9636 | 0.9725 | tau=0.92. Paired Δ=+0.0040 vs Exp3 (t=3.45, p<0.01; 252 S1s up / 105 down). FN 666→509, FP 175→170. India 0.9604 / US 0.9869. Top feats now: baddr_score, bbest_rank, bname_score. |

*Exp0 P/R equal the singleton rate by construction (1.0 on singletons, 0.0 elsewhere).
