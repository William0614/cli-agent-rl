#!/usr/bin/env python3
"""
Test script for RL Autotuner (dry-run mode)

This script tests the RL autotuner in dry-run mode to verify
the implementation without modifying system parameters.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cli_ai.tools.optimization.rl_autotuner import run_rl_optimization

def test_dry_run():
    """Test the RL autotuner in dry-run mode."""
    print("Testing RL Autotuner in Dry-Run Mode")
    print("=" * 80)
    
    # Path to example config
    config_path = os.path.join(
        os.path.dirname(__file__),
        'example_config.json'
    )
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    # Run optimization in dry-run mode
    results = run_rl_optimization(
        config_path=config_path,
        dry_run=True,
        verbose=True
    )
    
    # Check results
    if results['success']:
        print("\n✓ Test PASSED")
        print(f"Best reward: {results.get('best_reward', 'N/A')}")
        print(f"Best config: {results.get('best_config', 'N/A')}")
        return True
    else:
        print(f"\n✗ Test FAILED: {results.get('error', 'Unknown error')}")
        return False

if __name__ == '__main__':
    success = test_dry_run()
    sys.exit(0 if success else 1)
