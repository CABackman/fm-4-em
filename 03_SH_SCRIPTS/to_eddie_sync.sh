#!/usr/bin/env bash
set -euo pipefail

REMOTE_USERHOST="eddie"
REMOTE_BASE="/home/s1948145/FM-4-EM"

# Folders to sync (relative to where you run this script)
SRC_DIRS=(
  "01_DATASETS"
  "02_EMBEDDINGS"
  "03_SH_SCRIPTS"
  "05_PY_SCRIPTS"
  "08_CLASSIFICATION"
  "modules"
)

RSYNC_OPTS=(
  "-az"
  "--delete"                 # mirror: remove files on remote that were removed locally
  "--delete-delay"           # safer: delete at end of transfer
  "--info=progress2"
  "--human-readable"
  "--partial"
  "--exclude=.git/"
  "--exclude=__pycache__/"
  "--exclude=.ipynb_checkpoints/"
  "--exclude=*.pyc"
)

# Ensure remote base exists
ssh "${REMOTE_USERHOST}" "mkdir -p '${REMOTE_BASE}'"

# Sync each directory into REMOTE_BASE
for d in "${SRC_DIRS[@]}"; do
  rsync "${RSYNC_OPTS[@]}" "${d}/" "${REMOTE_USERHOST}:${REMOTE_BASE}/${d}/"
done
echo "Done syncing to ${REMOTE_USERHOST}:${REMOTE_BASE}"