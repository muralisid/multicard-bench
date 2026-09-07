#!/bin/zsh
cd /Users/muralisid/github_other/multicard-bench
set -o pipefail
echo "[chain] retrieve start $(date)"
uv run mcb run part1_retrieve --corpus lme --arms ours_cheap,ours_cheap_norule,ours_sentence_norule,S5_noPGR,oracle_full,closed_book --max-usd 35 --workers 4 || { echo "[chain] retrieve FAILED"; exit 1; }
echo "[chain] qa start $(date)"
uv run mcb run part1_qa --corpus lme --arms ours_cheap,ours_cheap_norule,ours_sentence_norule,S5_noPGR,oracle_full,closed_book --readers reader_a,reader_b --workers 4 --max-usd 20 || { echo "[chain] qa FAILED"; exit 1; }
echo "[chain] report start $(date)"
uv run mcb run part1_report --graphiti-status dropped || { echo "[chain] report FAILED"; exit 1; }
echo "[chain] DONE $(date)"
