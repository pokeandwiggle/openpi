#!/bin/bash
# Launch a training run + checkpoint-sync sidecar as systemd units on this VM.
#
# systemd owns the processes (survives SSH/session exits) and restarts the
# trainer on failure; scripts/train.py --resume is idempotent (fresh dir =>
# train from scratch, existing checkpoints => resume, same wandb run).
#
# Usage: launch_train_vm.sh <config_name> <exp_name> [repo_id] [sync_step_regex]
#   repo_id is required for configs that leave data.repo_id unset (pi05_paw);
#   pass "" to skip it for configs with a baked-in dataset.
set -euo pipefail

CONFIG=$1
EXP=$2
REPO_ID=${3:-}
SYNC_REGEX=${4:-'.*'}
USER_NAME=$(whoami)
OPENPI=/home/$USER_NAME/openpi
CKPT_BASE=${CKPT_BASE:-/mnt/localssd/checkpoints}
GS_BASE=${GS_BASE:-gs://training-artifacts-prod}

REPO_ARG=""
[ -n "$REPO_ID" ] && REPO_ARG="--data.repo-id='$REPO_ID'"

mountpoint -q /mnt/localssd || { echo "FATAL: /mnt/localssd not mounted (rebuild RAID0 after reboot)"; exit 1; }

# Units are keyed by exp_name: with the single pi05_paw config, the config name
# no longer identifies a run.
sudo systemd-run --unit="train-$EXP" \
  --property=Restart=on-failure --property=RestartSec=60 \
  --uid="$USER_NAME" --property=WorkingDirectory="$OPENPI" \
  -E XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 -E PATH="/home/$USER_NAME/.local/bin:/usr/bin:/bin" \
  bash -lc "source /etc/profile.d/training-env.sh && uv run scripts/train.py '$CONFIG' --exp-name='$EXP' $REPO_ARG --checkpoint-base-dir='$CKPT_BASE' --resume"

sudo systemd-run --unit="ckpt-sync-$EXP" \
  --property=Restart=always --property=RestartSec=30 \
  --uid="$USER_NAME" \
  bash -lc "source /etc/profile.d/training-env.sh && $OPENPI/scripts/sync_checkpoints_to_gcs.sh '$CKPT_BASE/$CONFIG/$EXP' '$GS_BASE/$CONFIG/$EXP' '$SYNC_REGEX'"

echo "Started units: train-$EXP, ckpt-sync-$EXP"
echo "  logs:   journalctl -fu train-$EXP"
echo "  stop:   sudo systemctl stop train-$EXP ckpt-sync-$EXP"
