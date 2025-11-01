#!/usr/bin/env python3
"""
Test the real-time RL visualization dashboard with mock data.
This simulates RL training to demonstrate the dashboard features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import random
import numpy as np
from cli_ai.tools.optimization.visualization import create_dashboard

def simulate_rl_training():
    """Simulate RL training with realistic data patterns."""
    
    # Create dashboard
    param_names = [
        'vm.dirty_ratio',
        'vm.dirty_background_ratio', 
        'vm.swappiness',
        'net.core.somaxconn'
    ]
    
    print("="*80)
    print("RL VISUALIZATION DASHBOARD - SIMULATION TEST")
    print("="*80)
    print("\nInitializing dashboard with parameters:")
    for param in param_names:
        print(f"  - {param}")
    print("\nDashboard will show:")
    print("  ✓ Learning curve (reward over time)")
    print("  ✓ Performance vs Stability scatter plot")
    print("  ✓ Episode rewards bar chart")
    print("  ✓ Parameter exploration heatmap")
    print("  ✓ Best configuration tracker")
    print("  ✓ Real-time statistics")
    print("\n" + "="*80 + "\n")
    
    dashboard = create_dashboard(param_names)
    dashboard.start()
    
    # Start update loop
    import threading
    update_thread = threading.Thread(target=dashboard.run_update_loop, daemon=True)
    update_thread.start()
    
    print("Dashboard initialized! Starting simulation...\n")
    time.sleep(2)
    
    # Simulate training
    baseline_performance = 1000.0
    baseline_stability = 50.0
    
    step = 0
    episode = 0
    steps_in_episode = 0
    max_steps_per_episode = 20
    
    try:
        for iteration in range(200):  # Simulate 200 steps
            step += 1
            steps_in_episode += 1
            
            # Simulate learning: performance gradually improves
            learning_progress = min(iteration / 100.0, 1.0)
            
            # Performance improves with some noise
            performance = baseline_performance * (1.0 + 0.5 * learning_progress + random.uniform(-0.1, 0.1))
            
            # Stability improves slightly
            stability = baseline_stability + 20 * learning_progress + random.uniform(-5, 5)
            
            # Calculate combined reward
            reward = 0.5 * performance + 0.5 * stability
            
            # Generate random parameter values (simulating exploration)
            params = {
                'vm.dirty_ratio': random.randint(10, 80),
                'vm.dirty_background_ratio': random.randint(5, 40),
                'vm.swappiness': random.randint(0, 60),
                'net.core.somaxconn': random.randint(512, 4096)
            }
            
            # Add data point to dashboard
            dashboard.add_data_point(
                step=step,
                reward=reward,
                performance=performance,
                stability=stability,
                episode=episode,
                params=params
            )
            
            # Print progress occasionally
            if step % 20 == 0:
                print(f"Step {step:3d} | Episode {episode:2d} | "
                      f"Reward: {reward:6.1f} | Perf: {performance:6.1f} | "
                      f"Stab: {stability:5.1f} | Best: {dashboard.best_reward:6.1f}")
            
            # New episode
            if steps_in_episode >= max_steps_per_episode:
                episode += 1
                steps_in_episode = 0
                if step % 40 == 0:
                    print(f"\n--- Episode {episode} complete ---\n")
            
            # Slow down simulation for visualization
            time.sleep(0.1)
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETE!")
        print("="*80)
        print(f"\nBest Configuration Found:")
        print(f"  Step: {dashboard.best_step}")
        print(f"  Reward: {dashboard.best_reward:.2f}")
        print(f"\nParameters:")
        for param, value in dashboard.best_config.items():
            print(f"  {param}: {value}")
        print("\n" + "="*80)
        print("\nDashboard will remain open for 30 seconds...")
        print("Observe the final visualizations:")
        print("  - Learning curve shows reward improvement over time")
        print("  - Performance vs Stability shows exploration pattern")
        print("  - Episode rewards show consistency")
        print("  - Parameter heatmap shows exploration strategy")
        print("\nClose the window manually or wait for auto-close.")
        print("="*80 + "\n")
        
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    finally:
        dashboard.stop()
        print("\nDashboard closed.")

if __name__ == "__main__":
    print("\nStarting RL visualization dashboard test...\n")
    
    try:
        simulate_rl_training()
        print("\n✅ Test completed successfully!")
        print("\nThis demonstrates what you'll see during actual RL optimization:")
        print("  1. Real-time learning curve")
        print("  2. Performance/stability trade-offs")
        print("  3. Parameter exploration patterns")
        print("  4. Best configuration tracking")
        print("\nThe actual optimization will show:")
        print("  - Real benchmark results instead of simulated data")
        print("  - Actual kernel parameter values")
        print("  - True performance improvements")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure matplotlib is installed:")
        print("  pip install matplotlib")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
