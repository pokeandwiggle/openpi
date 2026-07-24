#!/bin/bash
# Continuously sync completed orbax checkpoints to GCS.
#
# Watches <local_dir> for step directories that contain _CHECKPOINT_METADATA
# (orbax writes it as part of the finalized checkpoint, so its presence means
# the checkpoint is complete) and copies each matching one to <gs_dest> exactly
# once. A .synced_<step> stamp file next to the checkpoints records what was
# copied.
#
# <step_regex> filters which steps to sync (default: all). With frequent saves
# and a keep_period, sync only the keepers + the final step, e.g. for
# save_interval=1000, keep_period=10000, num_train_steps=60000:
#   '^([0-9]+0000|59999)$'
#
# Usage: sync_checkpoints_to_gcs.sh <local_dir> <gs_dest> [step_regex]
set -euo pipefail

LOCAL_DIR=$1
GS_DEST=${2%/}
STEP_REGEX=${3:-'.*'}

echo "Watching $LOCAL_DIR -> $GS_DEST (steps matching: $STEP_REGEX)"
while true; do
  if [ -d "$LOCAL_DIR" ]; then
    for meta in "$LOCAL_DIR"/*/_CHECKPOINT_METADATA; do
      [ -e "$meta" ] || continue
      step_dir=$(dirname "$meta")
      step=$(basename "$step_dir")
      echo "$step" | grep -qE "$STEP_REGEX" || continue
      stamp="$LOCAL_DIR/.synced_$step"
      [ -e "$stamp" ] && continue
      echo "$(date -u +%FT%TZ) syncing step $step"
      if gcloud storage cp -r "$step_dir" "$GS_DEST/"; then
        touch "$stamp"
        echo "$(date -u +%FT%TZ) synced step $step"
      else
        echo "$(date -u +%FT%TZ) sync FAILED for step $step (will retry)" >&2
      fi
    done
    # wandb_id.txt is tiny and appears once; keep it fresh.
    if [ -e "$LOCAL_DIR/wandb_id.txt" ] && [ ! -e "$LOCAL_DIR/.synced_wandb_id" ]; then
      gcloud storage cp "$LOCAL_DIR/wandb_id.txt" "$GS_DEST/wandb_id.txt" && touch "$LOCAL_DIR/.synced_wandb_id"
    fi
  fi
  sleep 60
done
