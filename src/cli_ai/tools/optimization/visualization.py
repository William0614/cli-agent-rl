"""
Real-time RL Training Visualization Dashboard

Displays:
- Learning curve (reward over time)
- Performance vs stability trade-off
- Parameter exploration heatmap
- Best configuration tracker
- Episode statistics

Works on openEuler (uses matplotlib + threading for real-time updates)
"""

import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for Linux compatibility
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import threading
import time
from datetime import datetime


class RLVisualizationDashboard:
    """Real-time visualization dashboard for RL training progress."""
    
    def __init__(self, max_points: int = 1000, update_interval: int = 500):
        """
        Initialize the dashboard.
        
        Args:
            max_points: Maximum number of data points to display
            update_interval: Update interval in milliseconds
        """
        self.max_points = max_points
        self.update_interval = update_interval
        
        # Data storage
        self.rewards = deque(maxlen=max_points)
        self.steps = deque(maxlen=max_points)
        self.performance_scores = deque(maxlen=max_points)
        self.stability_scores = deque(maxlen=max_points)
        self.episodes = deque(maxlen=max_points)
        
        # Best config tracking
        self.best_reward = -np.inf
        self.best_config = {}
        self.best_step = 0
        
        # Parameter exploration (store last N parameter sets)
        self.param_history = []
        self.param_names = []
        
        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        
        # Figure setup
        self.fig = None
        self.axes = {}
        
    def initialize(self, param_names: List[str]):
        """Initialize the visualization with parameter names."""
        self.param_names = param_names
        self._setup_figure()
        
    def _setup_figure(self):
        """Set up the matplotlib figure with subplots."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('RL Optimization Training Dashboard', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Learning Curve (top-left, large)
        self.axes['learning'] = self.fig.add_subplot(gs[0, :2])
        self.axes['learning'].set_title('Learning Curve: Reward over Time', fontweight='bold')
        self.axes['learning'].set_xlabel('Step')
        self.axes['learning'].set_ylabel('Reward')
        self.axes['learning'].grid(True, alpha=0.3)
        
        # 2. Best Config Display (top-right)
        self.axes['best_config'] = self.fig.add_subplot(gs[0, 2])
        self.axes['best_config'].set_title('Best Configuration', fontweight='bold')
        self.axes['best_config'].axis('off')
        
        # 3. Performance vs Stability (middle-left)
        self.axes['tradeoff'] = self.fig.add_subplot(gs[1, 0])
        self.axes['tradeoff'].set_title('Performance vs Stability', fontweight='bold')
        self.axes['tradeoff'].set_xlabel('Performance Score')
        self.axes['tradeoff'].set_ylabel('Stability Score')
        self.axes['tradeoff'].grid(True, alpha=0.3)
        
        # 4. Episode Rewards (middle-center)
        self.axes['episodes'] = self.fig.add_subplot(gs[1, 1])
        self.axes['episodes'].set_title('Episode Rewards', fontweight='bold')
        self.axes['episodes'].set_xlabel('Episode')
        self.axes['episodes'].set_ylabel('Avg Reward')
        self.axes['episodes'].grid(True, alpha=0.3)
        
        # 5. Parameter Exploration (middle-right)
        self.axes['params'] = self.fig.add_subplot(gs[1, 2])
        self.axes['params'].set_title('Parameter Exploration', fontweight='bold')
        
        # 6. Statistics Panel (bottom, spans all columns)
        self.axes['stats'] = self.fig.add_subplot(gs[2, :])
        self.axes['stats'].set_title('Training Statistics', fontweight='bold')
        self.axes['stats'].axis('off')
        
        plt.ion()  # Interactive mode
        self.fig.show()
        
    def add_data_point(self, step: int, reward: float, performance: float, 
                       stability: float, episode: int, params: Dict[str, float]):
        """Add a new data point to the dashboard."""
        with self.lock:
            self.steps.append(step)
            self.rewards.append(reward)
            self.performance_scores.append(performance)
            self.stability_scores.append(stability)
            self.episodes.append(episode)
            
            # Track parameter exploration
            self.param_history.append(params.copy())
            if len(self.param_history) > 100:  # Keep last 100
                self.param_history.pop(0)
            
            # Update best config
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_config = params.copy()
                self.best_step = step
    
    def update_plot(self, frame=None):
        """Update all plots with current data."""
        if not self.running:
            return
            
        with self.lock:
            if len(self.steps) == 0:
                return
                
            # Convert to arrays for plotting
            steps_arr = np.array(self.steps)
            rewards_arr = np.array(self.rewards)
            perf_arr = np.array(self.performance_scores)
            stab_arr = np.array(self.stability_scores)
            
            # 1. Update Learning Curve
            ax = self.axes['learning']
            ax.clear()
            ax.plot(steps_arr, rewards_arr, 'b-', alpha=0.7, linewidth=1, label='Reward')
            
            # Add moving average
            if len(rewards_arr) > 20:
                window = 20
                moving_avg = np.convolve(rewards_arr, np.ones(window)/window, mode='valid')
                ax.plot(steps_arr[window-1:], moving_avg, 'r-', linewidth=2, label='MA(20)')
            
            # Mark best point
            if self.best_step in steps_arr:
                idx = list(steps_arr).index(self.best_step)
                ax.plot(self.best_step, rewards_arr[idx], 'g*', markersize=15, 
                       label=f'Best: {self.best_reward:.2f}')
            
            ax.set_title('Learning Curve: Reward over Time', fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            # 2. Update Best Config Display
            ax = self.axes['best_config']
            ax.clear()
            ax.axis('off')
            
            config_text = f"Step: {self.best_step}\n"
            config_text += f"Reward: {self.best_reward:.2f}\n\n"
            config_text += "Parameters:\n"
            for param, value in self.best_config.items():
                # Shorten parameter names for display
                short_name = param.split('.')[-1]
                config_text += f"  {short_name}: {value}\n"
            
            ax.text(0.05, 0.95, config_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   family='monospace')
            
            # 3. Update Performance vs Stability
            ax = self.axes['tradeoff']
            ax.clear()
            
            # Scatter plot with color gradient
            scatter = ax.scatter(perf_arr, stab_arr, c=steps_arr, 
                               cmap='viridis', alpha=0.6, s=30)
            
            # Mark best point
            if len(perf_arr) > 0:
                best_idx = np.argmax(rewards_arr)
                ax.plot(perf_arr[best_idx], stab_arr[best_idx], 'r*', 
                       markersize=15, label='Best')
            
            ax.set_title('Performance vs Stability', fontweight='bold')
            ax.set_xlabel('Performance Score')
            ax.set_ylabel('Stability Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Update Episode Rewards
            if len(self.episodes) > 0:
                ax = self.axes['episodes']
                ax.clear()
                
                # Group by episode and calculate average
                unique_episodes = sorted(set(self.episodes))
                episode_avg_rewards = []
                
                for ep in unique_episodes:
                    ep_rewards = [r for e, r in zip(self.episodes, self.rewards) if e == ep]
                    if ep_rewards:
                        episode_avg_rewards.append(np.mean(ep_rewards))
                
                if episode_avg_rewards:
                    ax.bar(unique_episodes, episode_avg_rewards, alpha=0.7, color='skyblue')
                    ax.set_title('Episode Rewards', fontweight='bold')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Avg Reward')
                    ax.grid(True, alpha=0.3, axis='y')
            
            # 5. Update Parameter Exploration Heatmap
            if len(self.param_history) > 10 and len(self.param_names) > 0:
                ax = self.axes['params']
                ax.clear()
                
                # Create heatmap of parameter values over time
                param_matrix = []
                for params in self.param_history[-50:]:  # Last 50 steps
                    row = [params.get(p, 0) for p in self.param_names[:5]]  # Max 5 params
                    param_matrix.append(row)
                
                if param_matrix:
                    im = ax.imshow(np.array(param_matrix).T, aspect='auto', 
                                  cmap='RdYlGn', interpolation='nearest')
                    ax.set_yticks(range(len(self.param_names[:5])))
                    ax.set_yticklabels([p.split('.')[-1][:10] for p in self.param_names[:5]], 
                                      fontsize=8)
                    ax.set_xlabel('Recent Steps')
                    ax.set_title('Parameter Exploration', fontweight='bold')
                    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            
            # 6. Update Statistics Panel
            ax = self.axes['stats']
            ax.clear()
            ax.axis('off')
            
            stats_text = f"Steps: {steps_arr[-1]:,}  |  "
            stats_text += f"Episodes: {len(set(self.episodes))}  |  "
            stats_text += f"Current Reward: {rewards_arr[-1]:.2f}  |  "
            stats_text += f"Best Reward: {self.best_reward:.2f}  |  "
            
            if len(rewards_arr) > 1:
                improvement = ((rewards_arr[-1] - rewards_arr[0]) / max(abs(rewards_arr[0]), 1)) * 100
                stats_text += f"Improvement: {improvement:+.1f}%  |  "
            
            stats_text += f"Avg Performance: {np.mean(perf_arr):.2f}  |  "
            stats_text += f"Avg Stability: {np.mean(stab_arr):.2f}"
            
            ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
                   fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
        # Redraw
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except:
            pass  # Ignore errors during drawing
    
    def start(self):
        """Start the dashboard update loop."""
        self.running = True
        
    def stop(self):
        """Stop the dashboard and close the figure."""
        self.running = False
        plt.close(self.fig)
        
    def run_update_loop(self):
        """Run the update loop in a separate thread."""
        while self.running:
            try:
                self.update_plot()
                time.sleep(self.update_interval / 1000.0)
            except Exception as e:
                print(f"Dashboard update error: {e}")
                break


def create_dashboard(param_names: List[str]) -> RLVisualizationDashboard:
    """
    Create and initialize a visualization dashboard.
    
    Args:
        param_names: List of parameter names being optimized
        
    Returns:
        Initialized dashboard instance
    """
    dashboard = RLVisualizationDashboard()
    dashboard.initialize(param_names)
    return dashboard
