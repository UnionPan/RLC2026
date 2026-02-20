#!/bin/bash
# Unified training script for all environments and algorithms
#
# Usage: bash scripts/train.sh [ENVIRONMENT] [ALGORITHM] [CONFIG]
#
# Arguments:
#   ENVIRONMENT: competitive_fourrooms, competitive_obstructedmaze,
#                cooperative_keycorridor, cooperative_lavacrossing, pursuit_evasion
#   ALGORITHM:   reinforce, a2c, ppo
#   CONFIG:      quick, standard, long (optional, default: standard)

set -e  # Exit on error

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

ENV=${1:-"competitive_fourrooms"}
ALGO=${2:-"a2c"}
CONFIG_PRESET=${3:-"standard"}

# Validate environment
VALID_ENVS="competitive_fourrooms competitive_obstructedmaze cooperative_keycorridor cooperative_lavacrossing pursuit_evasion"
if [[ ! " $VALID_ENVS " =~ " $ENV " ]]; then
    echo "Error: Invalid environment '$ENV'"
    echo "Valid environments: $VALID_ENVS"
    exit 1
fi

# Validate algorithm
VALID_ALGOS="reinforce a2c ppo"
if [[ ! " $VALID_ALGOS " =~ " $ALGO " ]]; then
    echo "Error: Invalid algorithm '$ALGO'"
    echo "Valid algorithms: $VALID_ALGOS"
    exit 1
fi

# =============================================================================
# ALGORITHM-SPECIFIC CONFIG CHUNKS
# =============================================================================

# REINFORCE config
if [ "$ALGO" = "reinforce" ]; then
    case "$CONFIG_PRESET" in
        quick)
            EPISODES=100
            LR=1e-3
            UPDATE_FREQ=1
            ;;
        standard)
            EPISODES=3000
            LR=1e-3
            UPDATE_FREQ=1
            ;;
        long)
            EPISODES=10000
            LR=5e-4
            UPDATE_FREQ=1
            ;;
    esac
    GAMMA=0.99
    ENTROPY_COEF=0.01

# A2C config
elif [ "$ALGO" = "a2c" ]; then
    case "$CONFIG_PRESET" in
        quick)
            EPISODES=100
            LR=3e-4
            UPDATE_FREQ=5
            ;;
        standard)
            EPISODES=2000
            LR=3e-4
            UPDATE_FREQ=10
            ;;
        long)
            EPISODES=5000
            LR=1e-4
            UPDATE_FREQ=10
            ;;
    esac
    GAMMA=0.99
    ENTROPY_COEF=0.01

# PPO config
elif [ "$ALGO" = "ppo" ]; then
    case "$CONFIG_PRESET" in
        quick)
            EPISODES=100
            LR=3e-4
            UPDATE_FREQ=5
            ;;
        standard)
            EPISODES=2000
            LR=3e-4
            UPDATE_FREQ=10
            ;;
        long)
            EPISODES=5000
            LR=1e-4
            UPDATE_FREQ=10
            ;;
    esac
    GAMMA=0.99
    ENTROPY_COEF=0.01

fi

# =============================================================================
# GENERAL SETTINGS
# =============================================================================

DEVICE="cuda"  # GPU default with auto-fallback to CPU
SEED=42
EVAL_FREQ=100
SAVE_FREQ=200
SAVE_PATH="./checkpoints/${ENV}_${ALGO}"

# Create checkpoint directory
mkdir -p $SAVE_PATH

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "======================================================================"
echo "MULTI-AGENT RL TRAINING"
echo "======================================================================"
echo "Environment:    $ENV"
echo "Algorithm:      $ALGO"
echo "Config Preset:  $CONFIG_PRESET"
echo "======================================================================"
echo "Episodes:       $EPISODES"
echo "Learning Rate:  $LR"
echo "Gamma:          $GAMMA"
echo "Update Freq:    $UPDATE_FREQ"
echo "Device:         $DEVICE (auto-fallback to CPU if unavailable)"
echo "Seed:           $SEED"
echo "Save Path:      $SAVE_PATH"
echo "======================================================================"
echo ""

# =============================================================================
# RUN TRAINING
# =============================================================================

python -u src/main.py \
    --env $ENV \
    --algo $ALGO \
    --episodes $EPISODES \
    --lr $LR \
    --gamma $GAMMA \
    --device $DEVICE \
    --seed $SEED \
    2>&1 | tee "${SAVE_PATH}/training.log"

echo ""
echo "======================================================================"
echo "TRAINING COMPLETED!"
echo "======================================================================"
echo "Checkpoints saved in: $SAVE_PATH"
echo "Training log: ${SAVE_PATH}/training.log"
echo ""
echo "To test the trained agent, run:"
echo "  bash scripts/test.sh $ENV $ALGO"
echo "======================================================================"
