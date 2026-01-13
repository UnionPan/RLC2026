#!/bin/bash
# Unified testing/evaluation script for all environments and algorithms
#
# Usage: bash scripts/test.sh [ENVIRONMENT] [ALGORITHM] [OPTIONS]
#
# Arguments:
#   ENVIRONMENT: competitive_fourrooms, competitive_obstructedmaze,
#                cooperative_keycorridor, cooperative_lavacrossing, pursuit_evasion
#   ALGORITHM:   reinforce, a2c, ppo, mappo, qmix
#
# Options:
#   --episodes N     Number of evaluation episodes (default: 100)
#   --render MODE    Render mode: human, ansi, none (default: none)
#   --device DEVICE  Device: cuda, cpu (default: cuda)
#   --checkpoint PATH  Specific checkpoint path (default: latest in save_path)

set -e  # Exit on error

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

ENV=${1:-"competitive_fourrooms"}
ALGO=${2:-"a2c"}

# Default options
EPISODES=100
RENDER="none"
DEVICE="cuda"
CHECKPOINT=""

# Parse optional arguments
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --render)
            RENDER="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
VALID_ENVS="competitive_fourrooms competitive_obstructedmaze cooperative_keycorridor cooperative_lavacrossing pursuit_evasion"
if [[ ! " $VALID_ENVS " =~ " $ENV " ]]; then
    echo "Error: Invalid environment '$ENV'"
    echo "Valid environments: $VALID_ENVS"
    exit 1
fi

# Validate algorithm
VALID_ALGOS="reinforce a2c ppo mappo qmix"
if [[ ! " $VALID_ALGOS " =~ " $ALGO " ]]; then
    echo "Error: Invalid algorithm '$ALGO'"
    echo "Valid algorithms: $VALID_ALGOS"
    exit 1
fi

# =============================================================================
# FIND CHECKPOINT
# =============================================================================

SAVE_PATH="./checkpoints/${ENV}_${ALGO}"

if [ -z "$CHECKPOINT" ]; then
    # Find latest checkpoint automatically
    LATEST_CHECKPOINT=$(ls -t ${SAVE_PATH}/*_ep*.pt 2>/dev/null | head -1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "======================================================================"
        echo "ERROR: No checkpoint found!"
        echo "======================================================================"
        echo "Expected location: $SAVE_PATH"
        echo ""
        echo "Please train the model first:"
        echo "  bash scripts/train.sh $ENV $ALGO"
        echo "======================================================================"
        exit 1
    fi

    CHECKPOINT=$LATEST_CHECKPOINT
fi

# Extract episode number from checkpoint filename
EPISODE_NUM=$(basename $CHECKPOINT | sed -n 's/.*_ep\([0-9]*\)\.pt/\1/p')

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "======================================================================"
echo "MULTI-AGENT RL EVALUATION"
echo "======================================================================"
echo "Environment:    $ENV"
echo "Algorithm:      $ALGO"
echo "Checkpoint:     $CHECKPOINT"
echo "Trained Episode: $EPISODE_NUM"
echo "======================================================================"
echo "Eval Episodes:  $EPISODES"
echo "Render Mode:    $RENDER"
echo "Device:         $DEVICE"
echo "======================================================================"
echo ""

# =============================================================================
# CREATE EVALUATION SCRIPT
# =============================================================================

cat > /tmp/eval_${ENV}_${ALGO}.py << 'EVAL_SCRIPT_EOF'
#!/usr/bin/env python3
"""Dynamically generated evaluation script."""

import sys
import os
import torch
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.grid import competitive_envs, cooperatifve_envs, pursuit_evasion
from src.info_states import CNNLSTMInfoState, LSTMInfoState
from src.pg_agents import REINFORCEAgent, A2CAgent, PPOAgent

def create_env(env_name, render_mode=None):
    """Create environment."""
    if env_name == 'competitive_fourrooms':
        return competitive_envs.competitive_fourrooms_env(
            width=11, height=11, n_agents=2, view_radius=2,
            max_steps=200, render_mode=render_mode
        )
    elif env_name == 'competitive_obstructedmaze':
        return competitive_envs.competitive_obstructedmaze_env(
            width=11, height=11, n_agents=2, view_radius=2,
            max_steps=200, wall_density=0.1, render_mode=render_mode
        )
    elif env_name == 'cooperative_keycorridor':
        return cooperatifve_envs.coop_keycorridor_env(
            width=11, height=11, n_agents=2, view_radius=2,
            max_steps=200, render_mode=render_mode
        )
    elif env_name == 'cooperative_lavacrossing':
        return cooperatifve_envs.coop_lavacrossing_env(
            width=11, height=11, n_agents=2, view_radius=2,
            max_steps=200, render_mode=render_mode
        )
    elif env_name == 'pursuit_evasion':
        return pursuit_evasion.pursuit_evasion_env(
            width=15, height=15, n_pursuers=2, view_radius=3, max_steps=100
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def create_agent(env, algo, device='cpu'):
    """Create agent based on algorithm."""
    sample_agent = env.possible_agents[0]
    obs_shape = env.observation_space(sample_agent).shape
    action_dim = env.action_space(sample_agent).n

    # Create info state encoder
    encoder = CNNLSTMInfoState(
        obs_shape=obs_shape,
        action_dim=action_dim,
        state_dim=64,
        cnn_channels=[16, 32, 32],
    ).to(device)

    # Create agent
    agent_cfg = {'device': device}

    if algo == 'reinforce':
        agent = REINFORCEAgent(encoder, action_dim, agent_cfg)
    elif algo == 'a2c':
        agent = A2CAgent(encoder, action_dim, agent_cfg)
    elif algo == 'ppo':
        obs_space = env.observation_space(sample_agent)
        act_space = env.action_space(sample_agent)
        agent = PPOAgent(obs_space, act_space, encoder, agent_cfg)
    else:
        raise ValueError(f"Algorithm {algo} not supported in test script yet")

    return agent

def evaluate(env_name, algo, checkpoint_path, n_episodes=100, render_mode=None, device='cpu'):
    """Evaluate trained agent."""

    # Create environment
    env = create_env(env_name, render_mode)
    env.reset()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")

    # Create agents
    agents = {}
    for agent_id in env.possible_agents:
        agent = create_agent(env, algo, device)
        # Load weights if available
        if 'agent_state_dict' in checkpoint:
            try:
                agent.load_state_dict(checkpoint['agent_state_dict'])
            except:
                print(f"Warning: Could not load state dict for {agent_id}")
        agents[agent_id] = agent

    print(f"\nEvaluating for {n_episodes} episodes...")

    # Evaluate
    episode_rewards = {agent_id: [] for agent_id in env.possible_agents}
    episode_lengths = []
    win_counts = {agent_id: 0 for agent_id in env.possible_agents}

    for ep in range(n_episodes):
        env.reset()
        hiddens = {agent_id: agents[agent_id].info_state_encoder.init_hidden(1, device)
                   for agent_id in env.possible_agents}

        ep_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
        ep_length = 0

        for agent_id in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                action = None
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    agent = agents[agent_id]
                    if hasattr(agent, 'get_action'):
                        result = agent.get_action(obs_tensor, hiddens[agent_id], deterministic=True)
                        if len(result) == 4:
                            action, _, _, hiddens[agent_id] = result
                        else:
                            action, _, hiddens[agent_id] = result
                        action = action.item()
                    elif hasattr(agent, 'select_action'):
                        action, info_dict = agent.select_action(obs_tensor, deterministic=True)
                        hiddens[agent_id] = info_dict.get('hidden', None)
                    else:
                        action = env.action_space(agent_id).sample()

                ep_rewards[agent_id] += reward

            env.step(action)
            ep_length += 1

            if render_mode == 'human':
                env.render()

        # Track results
        for agent_id in ep_rewards:
            episode_rewards[agent_id].append(ep_rewards[agent_id])
            if ep_rewards[agent_id] > 0:
                win_counts[agent_id] += 1

        episode_lengths.append(ep_length)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} completed")

    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for agent_id in episode_rewards:
        mean_reward = np.mean(episode_rewards[agent_id])
        std_reward = np.std(episode_rewards[agent_id])
        win_rate = win_counts[agent_id] / n_episodes * 100
        print(f"{agent_id}:")
        print(f"  Mean reward:  {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"  Win rate:     {win_rate:.1f}%")

    print(f"\nMean episode length: {np.mean(episode_lengths):.2f}")
    print("="*60)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--render', type=str, default='none')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    render_mode = None if args.render == 'none' else args.render
    evaluate(args.env, args.algo, args.checkpoint, args.episodes, render_mode, args.device)
EVAL_SCRIPT_EOF

# =============================================================================
# RUN EVALUATION
# =============================================================================

python /tmp/eval_${ENV}_${ALGO}.py \
    --env "$ENV" \
    --algo "$ALGO" \
    --checkpoint "$CHECKPOINT" \
    --episodes $EPISODES \
    --render $RENDER \
    --device $DEVICE

echo ""
echo "======================================================================"
echo "EVALUATION COMPLETED!"
echo "======================================================================"
