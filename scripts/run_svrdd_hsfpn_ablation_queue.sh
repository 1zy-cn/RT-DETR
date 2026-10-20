#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The existing SCSA+HSFPN control occupies the same GPU. Pass its trainer PID
# to wait for it without interrupting or modifying its run directory.
control_pid="${1:-}"
if [[ -n "$control_pid" ]]; then
    while kill -0 "$control_pid" 2>/dev/null; do
        sleep 120
    done
    control_dir='runs/rtdetr-resnet18-HSFPN-SCSA-b8-s0'
    test -s "$control_dir/weights/last.pt"
    awk -F, 'END {gsub(/[[:space:]]/, "", $1); exit ($1 == 300 ? 0 : 1)}' \
        "$control_dir/results.csv"
fi

for experiment in hsfpn-scsa-dysample hsfpn-scsa-p2 hsfpn-scsa-dysample-p2; do
    case "$experiment" in
        hsfpn-scsa-dysample) run='rtdetr-resnet18-HSFPN-SCSA-DySample-b8-s0' ;;
        hsfpn-scsa-p2) run='rtdetr-resnet18-HSFPN-SCSA-P2-b8-s0' ;;
        hsfpn-scsa-dysample-p2) run='rtdetr-resnet18-HSFPN-SCSA-DySample-P2-b8-s0' ;;
    esac
    if [[ -e "runs/$run" ]]; then
        echo "Run directory already exists: runs/$run" >&2
        exit 1
    fi
    python train.py --experiment "$experiment" --check
    python train.py --experiment "$experiment"
    test -s "runs/$run/weights/last.pt"
    awk -F, 'END {gsub(/[[:space:]]/, "", $1); exit ($1 == 300 ? 0 : 1)}' \
        "runs/$run/results.csv"
done
