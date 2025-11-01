# SEAL-Inspired RL Autotuner

## Overview

The RL Autotuner is the "Tactician" layer of the SEAL-inspired Agentic Tuner system. It uses Reinforcement Learning (PPO algorithm) to automatically discover optimal kernel parameter configurations for specific workloads on openEuler OS.

## Architecture

```
User Request → LLM Strategist → JSON Config → RL Autotuner (Tactician) → Optimal Config
```

The RL Autotuner implements a custom Gymnasium environment that:
1. **Applies** kernel parameter changes via `sysctl`
2. **Runs** performance benchmarks
3. **Measures** system metrics (CPU, memory, I/O)
4. **Learns** optimal configurations through trial-and-error

## Features

### Safety Mechanisms
- **Parameter Validation**: Checks values against known safe ranges before applying
- **Rollback Capability**: Automatically reverts to default parameters on failure
- **Dry-Run Mode**: Test optimization logic without modifying system
- **Failure Tracking**: Stops after 3 consecutive failures to prevent instability

### Reward System
- **Multi-Objective**: 50% performance + 50% system stability
- **Performance**: Measured from benchmark output (e.g., requests/sec, transactions/sec)
- **Stability**: Calculated from I/O wait and memory pressure metrics

### State Space
The RL agent observes both:
- **System Metrics**: CPU utilization, I/O wait, memory usage from `/proc/stat` and `/proc/meminfo`
- **Kernel Parameters**: Current values of parameters being tuned

## Configuration File Format

```json
{
  "workload_name": "Descriptive name",
  "reward_metric": "metric_name_from_benchmark_output",
  "benchmark_command": "shell command to run benchmark",
  "action_space": [
    {
      "param": "kernel.parameter.name",
      "min": 100,
      "max": 10000,
      "type": "int"
    }
  ],
  "state_space": [
    {
      "metric": "cpu_utilization",
      "source": "/proc/stat"
    }
  ],
  "training_config": {
    "total_timesteps": 10000,
    "max_steps_per_episode": 50,
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "benchmark_timeout": 120
  }
}
```

## Usage

### Standalone Execution

```bash
# Run optimization
python src/cli_ai/tools/optimization/rl_autotuner.py --config config.json

# Dry-run mode (simulation only)
python src/cli_ai/tools/optimization/rl_autotuner.py --config config.json --dry-run

# Quiet mode
python src/cli_ai/tools/optimization/rl_autotuner.py --config config.json --quiet
```

### Programmatic Usage

```python
from cli_ai.tools.optimization.rl_autotuner import run_rl_optimization

results = run_rl_optimization(
    config_path='path/to/config.json',
    dry_run=False,
    verbose=True
)

if results['success']:
    print(f"Best configuration: {results['best_config']}")
    print(f"Performance improvement: {results['improvement']}%")
```

## Common Workload Examples

### Database Workload (PostgreSQL)

```json
{
  "workload_name": "PostgreSQL OLTP",
  "reward_metric": "transactions_per_second",
  "benchmark_command": "pgbench -c 50 -j 4 -T 30 testdb 2>&1",
  "action_space": [
    {"param": "vm.dirty_ratio", "min": 5, "max": 40, "type": "int"},
    {"param": "vm.dirty_background_ratio", "min": 1, "max": 20, "type": "int"},
    {"param": "vm.swappiness", "min": 0, "max": 10, "type": "int"}
  ]
}
```

### Web Server Workload (Nginx)

```json
{
  "workload_name": "Nginx High-Concurrency",
  "reward_metric": "requests_per_second",
  "benchmark_command": "wrk -t12 -c400 -d30s http://localhost:80/",
  "action_space": [
    {"param": "net.core.somaxconn", "min": 128, "max": 8192, "type": "int"},
    {"param": "net.ipv4.tcp_max_syn_backlog", "min": 128, "max": 8192, "type": "int"},
    {"param": "net.core.netdev_max_backlog", "min": 1000, "max": 50000, "type": "int"}
  ]
}
```

### HPC/Compute Workload

```json
{
  "workload_name": "CPU-Intensive Computation",
  "reward_metric": "operations_per_second",
  "benchmark_command": "sysbench cpu --time=30 run 2>&1",
  "action_space": [
    {"param": "kernel.sched_min_granularity_ns", "min": 100000, "max": 10000000, "type": "int"},
    {"param": "kernel.sched_wakeup_granularity_ns", "min": 100000, "max": 15000000, "type": "int"}
  ]
}
```

## Output Format

The autotuner provides real-time progress output:

```
================================================================================
RL Autotuner Environment Initialized
================================================================================
Workload: High-Throughput Web Server
Reward Metric: requests_per_second
Action Space: 3 parameters
State Space: 6 metrics
================================================================================

--- Step 1 ---
Action (parameters to apply):
  net.core.somaxconn: 2048
  net.core.netdev_max_backlog: 5000
  ✓ Applied net.core.somaxconn=2048
  ✓ Applied net.core.netdev_max_backlog=5000
  Running benchmark: ...
  ✓ Benchmark completed in 31.2s: requests_per_second=1523.5
  🎯 New best reward: 1523.5
  Reward: 1523.5 (perf: 1523.5, stab: 0.89)
  Improvement over baseline: +12.3%

...

================================================================================
Optimization Complete!
================================================================================
Workload: High-Throughput Web Server
Total Episodes: 15
Baseline Performance: 1356.2
Best Performance: 1789.3
Improvement: +31.9%

Optimal Configuration:
  net.core.somaxconn = 3584
  net.core.netdev_max_backlog = 7800

To apply this configuration permanently, run:
  sudo sysctl -w net.core.somaxconn=3584
  sudo sysctl -w net.core.netdev_max_backlog=7800
================================================================================
```

## Safety Considerations

1. **Always test in dry-run mode first** on production systems
2. **Benchmark commands should be non-destructive** and representative
3. **Parameter ranges should be conservative** - use known safe values
4. **Monitor system health** during optimization
5. **Have rollback plan ready** - default parameters are automatically stored

## Integration with Multimodal CLI Agent

The RL autotuner integrates with the main CLI agent as a tool:

1. User issues natural language command: `"optimize this system for database workload"`
2. LLM Strategist generates appropriate JSON configuration
3. CLI agent invokes RL autotuner with the configuration
4. Real-time progress is streamed to user terminal
5. Best configuration is presented and can be applied

## Troubleshooting

### Permission Errors
```bash
# Ensure sudo access for sysctl
sudo visudo
# Add: your_user ALL=(ALL) NOPASSWD: /usr/sbin/sysctl
```

### Benchmark Parsing Fails
- Check that `reward_metric` matches text in benchmark output
- Use regex-friendly metric names (avoid special characters)
- Test benchmark command manually first

### Poor Convergence
- Increase `total_timesteps` in config
- Adjust `learning_rate` (try 1e-4 to 1e-3)
- Ensure benchmark has low variance
- Check if parameter ranges are too wide

### System Instability
- Reduce parameter ranges
- Lower `max_steps_per_episode`
- Increase stability weight in code (currently 50/50)
- Use dry-run mode for testing

## Future Enhancements

- [ ] Support for discrete action spaces
- [ ] Multi-node distributed optimization
- [ ] Transfer learning between similar workloads
- [ ] Automated benchmark selection
- [ ] Integration with monitoring systems (Prometheus, Grafana)
- [ ] Catastrophic forgetting mitigation (SEAL-style)
- [ ] Meta-learning for faster adaptation
