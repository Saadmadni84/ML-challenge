#!/usr/bin/env bash
# Build the final submission zip AFTER the full Colab run.
# Validates outputs first, then packs the exact required structure:
#   <team>_submission.zip
#   |-- output/matching_results.tsv + output/candidate_pairs.tsv
#   |-- code/business_entity_resolution/{src,README.md,requirements.txt}
#   |-- Documentation_template.md (filled in)
# Usage: bash utils/make_submission_zip.sh <team_name>   (run from repo root)
set -euo pipefail
TEAM="${1:?usage: bash utils/make_submission_zip.sh <team_name>}"
test -f output/matching_results.tsv || { echo "missing output/matching_results.tsv (run notebook 02b first)"; exit 1; }
test -f output/candidate_pairs.tsv   || { echo "missing output/candidate_pairs.tsv (run notebook 02b first)"; exit 1; }
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
rm -f "${TEAM}_submission.zip"
zip -qr "${TEAM}_submission.zip" output/matching_results.tsv output/candidate_pairs.tsv \
  code/business_entity_resolution Documentation_template.md \
  -x '*/__pycache__/*' '*/.ipynb_checkpoints/*'
echo "PACKED: ${TEAM}_submission.zip"
unzip -l "${TEAM}_submission.zip" | tail -5
