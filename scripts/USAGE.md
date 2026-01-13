## **Unified Training & Testing Scripts**

All training and testing is done through **two simple scripts**:
- `train.sh` - Train agents
- `test.sh` - Evaluate trained agents

---

## **🚀 Quick Start**

### Train an Agent
```bash
# Basic usage: bash scripts/train.sh [ENVIRONMENT] [ALGORITHM] [CONFIG]
bash scripts/train.sh competitive_fourrooms a2c

# With config preset
bash scripts/train.sh competitive_fourrooms ppo standard
bash scripts/train.sh cooperative_keycorridor mappo long
```

### Test a Trained Agent
```bash
# Basic usage: bash scripts/test.sh [ENVIRONMENT] [ALGORITHM]
bash scripts/test.sh competitive_fourrooms a2c

# With visual rendering
bash scripts/test.sh competitive_fourrooms ppo --render human --episodes 10
```

---

## **📋 Arguments**

### **`train.sh` Arguments**

```bash
bash scripts/train.sh [ENVIRONMENT] [ALGORITHM] [CONFIG_PRESET]
```

**ENVIRONMENT** (required):
- `competitive_fourrooms` - Two agents race to reach goal
- `competitive_obstructedmaze` - Navigate maze competitively
- `cooperative_keycorridor` - Cooperatively collect key and reach goal
- `cooperative_lavacrossing` - Navigate lava together
- `pursuit_evasion` - Pursuers chase evader

**ALGORITHM** (required):
- `reinforce` - Basic policy gradient
- `a2c` - Advantage Actor-Critic (recommended for single-agent)
- `ppo` - Proximal Policy Optimization (best sample efficiency)
- `mappo` - Multi-Agent PPO (recommended for cooperation)
- `qmix` - Value factorization (for cooperation)

**CONFIG_PRESET** (optional, default: `standard`):
- `quick` - Fast testing (~100-200 episodes)
- `standard` - Normal training (~2000-5000 episodes)
- `long` - Extended training (~5000-10000 episodes)

---

### **`test.sh` Arguments**

```bash
bash scripts/test.sh [ENVIRONMENT] [ALGORITHM] [OPTIONS]
```

**OPTIONS**:
- `--episodes N` - Number of evaluation episodes (default: 100)
- `--render MODE` - Render mode: `human`, `ansi`, `none` (default: none)
- `--device DEVICE` - Device: `cuda`, `cpu` (default: cuda)
- `--checkpoint PATH` - Specific checkpoint file (default: latest)

---

## **📊 Algorithm Configs**

The scripts automatically apply algorithm-specific hyperparameters:

### **REINFORCE** (Policy Gradient)
- **Best for**: Simple baselines, small-scale experiments
- **Learning rate**: 1e-3
- **Episodes**: 3000 (standard)
- **Update freq**: Every 1 episode

### **A2C** (Actor-Critic)
- **Best for**: Competitive environments, fast training
- **Learning rate**: 3e-4
- **Episodes**: 2000 (standard)
- **Update freq**: Every 10 episodes
- **DEFAULT algorithm**

### **PPO** (Proximal Policy Optimization)
- **Best for**: Sample efficiency, stable training
- **Learning rate**: 3e-4
- **Episodes**: 2000 (standard)
- **Update freq**: Every 10 episodes
- **Clip epsilon**: 0.2
- **GAE lambda**: 0.95

### **MAPPO** (Multi-Agent PPO)
- **Best for**: Cooperative tasks (key corridor, lava crossing)
- **Learning rate**: 5e-4
- **Episodes**: 5000 (standard)
- **Update freq**: Every 20 episodes
- **Centralized critic** for better coordination

### **QMIX** (Value Decomposition)
- **Best for**: Cooperative tasks with shared rewards
- **Learning rate**: 5e-4
- **Episodes**: 5000 (standard)
- **Update freq**: Every 20 episodes
- **Monotonic mixing** for credit assignment

---

## **💡 Examples**

### Example 1: Train PPO on Competitive FourRooms
```bash
# Standard training (~2000 episodes, ~30 min on GPU)
bash scripts/train.sh competitive_fourrooms ppo

# Quick test (~100 episodes, ~2 min)
bash scripts/train.sh competitive_fourrooms ppo quick

# Long training (~5000 episodes, ~1 hour)
bash scripts/train.sh competitive_fourrooms ppo long
```

### Example 2: Train MAPPO on Cooperative Task
```bash
# MAPPO is best for cooperation
bash scripts/train.sh cooperative_keycorridor mappo standard

# Evaluate with visual rendering
bash scripts/test.sh cooperative_keycorridor mappo --render human
```

### Example 3: Compare Algorithms
```bash
# Train same environment with different algorithms
bash scripts/train.sh competitive_fourrooms a2c standard
bash scripts/train.sh competitive_fourrooms ppo standard
bash scripts/train.sh competitive_fourrooms reinforce standard

# Evaluate all
bash scripts/test.sh competitive_fourrooms a2c --episodes 100
bash scripts/test.sh competitive_fourrooms ppo --episodes 100
bash scripts/test.sh competitive_fourrooms reinforce --episodes 100
```

### Example 4: Pursuit-Evasion
```bash
# Train with A2C (good for asymmetric games)
bash scripts/train.sh pursuit_evasion a2c standard

# Evaluate and watch
bash scripts/test.sh pursuit_evasion a2c --render human --episodes 5
```

---

## **📁 Checkpoint Management**

### Automatic Checkpoint Saving

**Checkpoints are saved automatically** every 200 episodes to:
```
./checkpoints/[ENVIRONMENT]_[ALGORITHM]/
```

**Example structure:**
```
checkpoints/
├── competitive_fourrooms_a2c/
│   ├── training.log
│   ├── agent_0_ep200.pt
│   ├── agent_0_ep400.pt
│   └── ...
├── competitive_fourrooms_ppo/
│   ├── training.log
│   ├── agent_0_ep200.pt
│   └── ...
└── cooperative_keycorridor_mappo/
    └── ...
```

### Using Specific Checkpoints

```bash
# Test with specific checkpoint
bash scripts/test.sh competitive_fourrooms ppo \
    --checkpoint ./checkpoints/competitive_fourrooms_ppo/agent_0_ep2000.pt
```

---

## **🖥️ Device Selection**

**GPU is the default** with automatic CPU fallback:

```bash
# Uses GPU by default (auto-fallback to CPU)
bash scripts/train.sh competitive_fourrooms a2c

# Force CPU
# Edit scripts/train.sh: DEVICE="cpu"
```

**Performance:**
- GPU: ~30-60 min for 2000 episodes
- CPU: ~8-12 hours for 2000 episodes
- **Speedup: ~15-20x on GPU**

---

## **📝 Training Logs**

All training output is saved to `[checkpoint_dir]/training.log`:

```bash
# Watch training live
tail -f checkpoints/competitive_fourrooms_a2c/training.log

# View completed log
cat checkpoints/competitive_fourrooms_a2c/training.log
```

---

## **🎯 Best Practices**

### 1. **Start with Quick Tests**
```bash
# Always test with 'quick' config first
bash scripts/train.sh competitive_fourrooms a2c quick
bash scripts/test.sh competitive_fourrooms a2c
```

### 2. **Choose Algorithm by Task**

| Task Type | Best Algorithm | Why |
|-----------|---------------|-----|
| **Competitive** | A2C, PPO | Fast, stable, good for independent agents |
| **Cooperative** | MAPPO, QMIX | Centralized critic helps coordination |
| **Baseline** | REINFORCE | Simple, interpretable |

### 3. **Monitor Training**
```bash
# Open training log in another terminal
tail -f checkpoints/*/training.log
```

### 4. **Compare Algorithms**
```bash
# Train same env with multiple algorithms
for algo in a2c ppo mappo; do
    bash scripts/train.sh cooperative_keycorridor $algo standard
done

# Evaluate all
for algo in a2c ppo mappo; do
    bash scripts/test.sh cooperative_keycorridor $algo --episodes 100
done
```

---

## **🔧 Advanced Usage**

### Environment Variables

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh competitive_fourrooms ppo

# Parallel training on multiple GPUs
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh competitive_fourrooms a2c &
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh cooperative_keycorridor mappo &
```

### Direct Python API

```bash
# Bypass scripts for fine-grained control
python src/main.py \
    --env competitive_fourrooms \
    --algo ppo \
    --episodes 2000 \
    --lr 1e-4 \
    --device cuda
```

---

## **❓ Troubleshooting**

### "No checkpoint found"
```bash
# Train first
bash scripts/train.sh competitive_fourrooms a2c
```

### "Invalid environment/algorithm"
```bash
# Check valid options
bash scripts/train.sh  # Shows error with valid options
```

### "CUDA out of memory"
```bash
# Edit scripts/train.sh and set:
DEVICE="cpu"
```

---

## **📚 Summary**

**Training:**
```bash
bash scripts/train.sh [ENV] [ALGO] [CONFIG]
```

**Testing:**
```bash
bash scripts/test.sh [ENV] [ALGO] [OPTIONS]
```

**Example workflow:**
```bash
# 1. Quick test
bash scripts/train.sh competitive_fourrooms ppo quick

# 2. Full training
bash scripts/train.sh competitive_fourrooms ppo standard

# 3. Evaluate
bash scripts/test.sh competitive_fourrooms ppo --episodes 100

# 4. Watch agent play
bash scripts/test.sh competitive_fourrooms ppo --render human --episodes 5
```

**That's it! Two scripts for everything. 🚀**
