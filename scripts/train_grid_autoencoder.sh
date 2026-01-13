#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-competitive_fourrooms}"
OUTPUT_DIR="${2:-grid_autoencoder_runs}"
LATENT_DIM="${3:-64}"
EPISODES="${4:-300}"
EPOCHS="${5:-100}"

python -m src.info_states.grid_autoencoder.train \
  --env_name "${ENV_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim "${LATENT_DIM}" \
  --episodes "${EPISODES}" \
  --epochs "${EPOCHS}"
