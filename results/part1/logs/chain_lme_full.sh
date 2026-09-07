#!/bin/zsh
cd /Users/muralisid/github_other/multicard-bench
echo "[chain] retrieve full start $(date)"
uv run mcb run part1_retrieve --corpus lme --max-usd 35 --workers 4 || { echo "[chain] retrieve FAILED"; exit 1; }
echo "[chain] qa start $(date)"
uv run mcb run part1_qa --corpus all --readers reader_a,reader_b --workers 4 --max-usd 40 || { echo "[chain] qa FAILED"; exit 1; }
echo "[chain] report start $(date)"
uv run mcb run part1_report --graphiti-status dropped || { echo "[chain] report FAILED"; exit 1; }
echo "[chain] DONE $(date)"
