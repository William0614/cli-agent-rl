# RL-based OS Optimization System

A SEAL-inspired Reinforcement Learning autotuner for kernel parameter optimization, integrated into the multimodal CLI agent.

## 🎯 Overview

This system uses **Proximal Policy Optimization (PPO)** to automatically discover optimal operating system kernel parameter configurations for specific workloads. Inspired by Microsoft's SEAL framework, it treats OS tuning as a reinforcement learning problem where:

- **Agent**: PPO neural network learning optimal parameter values
- **Environment**: Your operating system with configurable kernel parameters
- **Actions**: Setting kernel parameters (e.g., `vm.swappiness`, `vm.dirty_ratio`)
- **Rewards**: Performance metrics from workload benchmarks
- **Goal**: Maximize workload performance while maintaining system stability

## 🚀 Key Features

### 🎓 Intelligent Learning
- **PPO Algorithm**: Stable policy gradient method for continuous action spaces
- **Adaptive Exploration**: Balances exploration (trying new configs) with exploitation (using known good configs)
- **Episode-based Training**: Each episode tests a parameter configuration and measures performance
- **Best Config Tracking**: Remembers the optimal configuration discovered during training

### 📊 Real-time Visualization
- **Web Dashboard**: Flask-based live dashboard accessible from any device on the network
- **Learning Curves**: Watch the agent improve over time
- **Performance Metrics**: Real-time reward, stability, and parameter tracking
- **Episode Analysis**: Per-episode reward distribution and trends

### 🛡️ Safety & Reliability
- **Parameter Validation**: Ensures values stay within safe operating ranges
- **Automatic Rollback**: Restores original system parameters on completion
- **Failure Handling**: Detects and recovers from benchmark failures
- **Dry-run Mode**: Test configurations without actually modifying the system

### ⚡ Performance Optimized
- **Fast Benchmarks**: Custom lightweight benchmarks (8-15 seconds each)
- **Efficient Training**: Typical optimization completes in 20-30 minutes
- **Parallel Updates**: Batch processing for stable learning
- **Early Stopping**: Terminates when optimal solution is found

## 📁 Architecture

```
RL Autotuner System
├── Core RL Engine
│   ├── rl_autotuner.py         # Main training loop, PPO integration
│   ├── OSTuningEnv              # Gymnasium environment
│   └── SafetyValidator          # Parameter range validation
│
├── Visualization
│   ├── web_dashboard.py         # Flask server + SSE streaming
│   └── templates/
│       └── rl_dashboard.html    # Real-time web interface
│
├── Benchmarks
│   ├── fast_memory_bench.py     # Memory + I/O workload
│   └── fast_network_bench.py    # Network performance workload
│
└── Configuration
    └── configs/
        └── hackathon_demo_config.json
```

## 🔧 How It Works

### 1. **Environment Setup**
The system creates a Gymnasium environment that:
- Reads current kernel parameter values (baseline)
- Defines action space (parameters to tune) and observation space (system metrics)
- Sets up benchmark command for performance measurement

### 2. **Training Loop**
For each training step:
```python
1. Agent selects action (parameter values) based on current policy
2. System applies parameters: sudo sysctl -w param=value
3. Benchmark runs and measures performance
4. Reward calculated: R = performance * 0.5 + stability * 0.5
5. Agent updates policy using PPO to maximize future rewards
```

### 3. **Policy Update (PPO)**
The PPO algorithm:
- Uses actor-critic architecture (policy + value function)
- Clips policy updates to prevent drastic changes
- Performs multiple epochs of mini-batch updates
- Balances exploration via entropy bonus

### 4. **Web Dashboard Updates**
Real-time streaming via Server-Sent Events (SSE):
- Dashboard polls `/data` endpoint every 2 seconds
- Learning curves and plots refresh every 5 seconds
- Auto-stops when training completes

## 🎮 Usage

### Quick Start

```bash
# Using the CLI agent
python main.py

> Optimize my system for memory-intensive workloads
```

The agent will:
1. Start the web dashboard on `http://0.0.0.0:5000`
2. Begin RL training with PPO
3. Display real-time progress
4. Report optimal configuration when complete

### Advanced Usage

```python
from src.cli_ai.tools.optimization.rl_autotuner import optimize_os_parameters

# Load configuration
config = {
    "workload_name": "memory-optimization",
    "reward_metric": "operations_per_second",
    "benchmark_command": "python3 benchmarks/fast_memory_bench.py",
    "action_space": [
        {"param": "vm.swappiness", "min": 0, "max": 100, "type": "int"},
        {"param": "vm.dirty_ratio", "min": 5, "max": 80, "type": "int"}
    ],
    "training_config": {
        "total_timesteps": 200,
        "learning_rate": 0.001,
        "benchmark_timeout": 15
    }
}

# Run optimization
result = optimize_os_parameters(
    config=config,
    show_dashboard=True,
    web_host="0.0.0.0",
    web_port=5000
)

print(f"Best config: {result['best_config']}")
print(f"Performance: {result['best_reward']:.2f}")
print(f"Improvement: {result['improvement']:.2f}%")
```

### Configuration Files

Create a JSON config in `configs/`:

```json
{
  "workload_name": "my-workload",
  "reward_metric": "operations_per_second",
  "benchmark_command": "python3 benchmarks/my_benchmark.py",
  "action_space": [
    {
      "param": "vm.swappiness",
      "min": 0,
      "max": 100,
      "type": "int",
      "description": "How aggressively kernel swaps memory pages"
    },
    {
      "param": "vm.dirty_ratio",
      "min": 5,
      "max": 80,
      "type": "int",
      "description": "Percentage of RAM for dirty pages before writes"
    }
  ],
  "state_space": [
    {
      "metric": "cpu_usage",
      "description": "Current CPU utilization percentage"
    },
    {
      "metric": "memory_used_percent",
      "description": "Memory utilization percentage"
    }
  ],
  "training_config": {
    "total_timesteps": 200,
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 5,
    "learning_rate": 0.001,
    "gamma": 0.95,
    "benchmark_timeout": 15,
    "max_steps_per_episode": 10,
    "verbose": false
  }
}
```

## 📊 Web Dashboard

### Accessing the Dashboard

Once training starts, open your browser to:
```
http://localhost:5000          # From the same machine
http://192.168.x.x:5000        # From another machine on the network
```

### Dashboard Features

**Status Bar**
- Training status (Training/Stopped/Complete)
- Total steps and episodes completed
- Current reward and best reward found
- Runtime elapsed

**Learning Curve**
- Reward over time (steps)
- Shows agent improvement
- Smoothed line for trend visualization

**Performance vs Stability**
- Scatter plot showing trade-off
- Helps identify balanced configurations

**Episode Rewards**
- Bar chart of average reward per episode
- Shows learning progress across episodes

**Best Configuration**
- Current best parameters found
- Reward achieved
- Step number when found

### Real-time Updates

The dashboard uses Server-Sent Events (SSE) for live streaming:
- Data refreshes automatically every 2 seconds
- Plots update every 5 seconds
- No manual refresh needed
- Stops updating when training completes

## 🔬 Creating Custom Benchmarks

### Benchmark Requirements

Your benchmark must:
1. **Be executable**: Shell command or Python script
2. **Output parseable metric**: Print `metric_name: value` format
3. **Complete quickly**: 5-30 seconds recommended
4. **Be deterministic**: Similar parameters → similar performance

### Example Benchmark Template

```python
#!/usr/bin/env python3
import subprocess
import sys

def get_vm_params():
    """Read current kernel parameters"""
    try:
        swappiness = int(subprocess.check_output(
            ['sysctl', '-n', 'vm.swappiness']
        ).decode().strip())
        return swappiness
    except:
        return 60  # default

def run_workload():
    """Execute your workload and measure performance"""
    # Your workload code here
    # Return a performance metric
    return performance_value

def main():
    # Get current parameters
    swappiness = get_vm_params()
    
    # Run workload
    performance = run_workload()
    
    # Output in parseable format
    print(f"operations_per_second: {performance:.2f}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

### PPO (Proximal Policy Optimization)

PPO is a policy gradient method that:

1. **Samples trajectories** under current policy
2. **Computes advantages** (how good each action was)
3. **Updates policy** to increase probability of good actions
4. **Clips updates** to prevent large policy changes

Key hyperparameters:
- `learning_rate`: Step size for policy updates (0.001)
- `gamma`: Discount factor for future rewards (0.95)
- `n_steps`: Steps per rollout batch (64)
- `batch_size`: Mini-batch size for updates (32)
- `n_epochs`: Optimization epochs per batch (5)

### Reward Function

The default reward combines two factors:

```python
reward = performance_weight * performance_score + 
         stability_weight * stability_score

# Default: 50% performance, 50% stability
performance_weight = 0.5
stability_weight = 0.5
```

**Performance Score**: Direct benchmark output (e.g., ops/sec)
**Stability Score**: System health metric (0-1 range)

### Training Process

```
Episode 1: Random params → Low performance → Negative reward
Episode 2: Slightly better params → Improved performance → Positive reward
Episode 3: Agent learns pattern → Tries optimal range → High reward
...
Episode N: Converges to optimal configuration
```

The agent learns that certain parameter ranges consistently yield higher rewards.

## 🛡️ Safety & Validation

### Parameter Validation

Before applying any parameter:
```python
SafetyValidator.validate(param_name, value, min_val, max_val)
```

Checks:
- ✅ Value is within configured range
- ✅ Parameter exists on the system
- ✅ Value type is correct (int/float)

### Automatic Rollback

The system automatically:
1. **Stores original values** before training
2. **Restores them** when training completes
3. **Handles failures** silently (best-effort restoration)

### Dry-run Mode

Test without system modification:
```python
result = optimize_os_parameters(
    config=config,
    dry_run=True  # Don't actually apply parameters
)
```

## 📈 Performance Tips

### Faster Training

1. **Reduce timesteps**: `total_timesteps: 100` for quick tests
2. **Shorter benchmarks**: Keep under 15 seconds
3. **Fewer episodes**: `max_steps_per_episode: 5`
4. **Larger batches**: `batch_size: 64` for faster convergence

### Better Results

1. **More timesteps**: `total_timesteps: 500+` for thorough exploration
2. **Sensitive benchmarks**: Ensure params actually affect performance
3. **Appropriate ranges**: Don't make action space too large
4. **Multiple runs**: Run 3-5 times and take best result

## 🐛 Troubleshooting

### Dashboard Not Loading

**Problem**: Can't access `http://localhost:5000`

**Solutions**:
```bash
# Check if port is in use
lsof -i :5000

# Use different port
# Modify web_port in optimization call

# Check firewall
sudo ufw allow 5000
```

### Benchmark Failing

**Problem**: "Benchmark failed with exit code 1"

**Solutions**:
```bash
# Test benchmark manually
python3 benchmarks/fast_memory_bench.py

# Check output format
# Should print: metric_name: value

# Increase timeout
"benchmark_timeout": 30
```

### No Learning Progress

**Problem**: Reward stays flat

**Solutions**:
1. Check if benchmark is parameter-sensitive
2. Verify parameter ranges are appropriate
3. Increase learning rate: `0.003`
4. Reduce batch size: `16`
5. Try different random seed

### Permission Errors

**Problem**: "sysctl: permission denied"

**Solutions**:
```bash
# Ensure sudo access
sudo -v

# Check sudo doesn't require password for sysctl
sudo sysctl -w vm.swappiness=60

# Run script with sudo if needed
```