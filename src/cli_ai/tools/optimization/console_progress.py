"""
Simple console-based progress tracker for RL training.
Used as fallback when dashboard visualization is not available.
"""

import sys
from typing import Dict
from collections import deque


class ConsoleProgressTracker:
    """Simple console-based progress display for RL training."""
    
    def __init__(self):
        self.best_reward = -float('inf')
        self.best_config = {}
        self.best_step = 0
        self.step_count = 0
        self.episode_count = 0
        self.recent_rewards = deque(maxlen=10)
        
    def update(self, step: int, reward: float, performance: float, 
               stability: float, episode: int, params: Dict[str, float]):
        """Update progress with new data point."""
        self.step_count = step
        self.episode_count = episode
        self.recent_rewards.append(reward)
        
        # Track best
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_config = params.copy()
            self.best_step = step
            
            # Print update when we find a better config
            print(f"\n🎯 New Best! Step {step}, Reward: {reward:.2f}")
            print(f"   Performance: {performance:.2f}, Stability: {stability:.2f}")
        
        # Print periodic updates
        if step % 100 == 0:
            avg_recent = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
            print(f"\n📊 Step {step:5d} | Episode {episode:3d} | "
                  f"Recent Avg: {avg_recent:6.1f} | Best: {self.best_reward:6.1f}")
    
    def print_summary(self):
        """Print final summary."""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"Total Steps: {self.step_count}")
        print(f"Total Episodes: {self.episode_count}")
        print(f"Best Reward: {self.best_reward:.2f} (at step {self.best_step})")
        print("\nBest Configuration:")
        for param, value in self.best_config.items():
            print(f"  {param}: {value}")
        print("="*80)
