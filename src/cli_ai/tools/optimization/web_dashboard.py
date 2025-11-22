"""
Web-based real-time visualization dashboard for RL training.
Streams updates to web browser - view from any machine on the network.

Usage:
    python -m src.cli_ai.tools.optimization.web_dashboard
    Then open: http://<vm-ip>:5000

Updates automatically via Server-Sent Events (SSE)
"""

from flask import Flask, render_template, Response, jsonify
import json
import time
import os
import logging
from datetime import datetime
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from threading import Lock
from pathlib import Path

# Suppress Flask request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Get the project root directory (4 levels up from this file)
# web_dashboard.py -> optimization -> tools -> cli_ai -> src -> root
project_root = Path(__file__).parent.parent.parent.parent.parent
template_dir = project_root / 'templates'

app = Flask(__name__, template_folder=str(template_dir))

# Global state (thread-safe)
state_lock = Lock()
training_state = {
    'steps': deque(maxlen=1000),
    'rewards': deque(maxlen=1000),
    'performance': deque(maxlen=1000),
    'stability': deque(maxlen=1000),
    'episodes': deque(maxlen=1000),
    'param_history': [],
    'best_reward': -float('inf'),
    'best_config': {},
    'best_step': 0,
    'is_training': False,
    'training_complete': False,  # New flag to stop updates
    'start_time': None,
    'workload_name': '',
    'param_names': [],
    'config': {}  # Store full optimization config
}

class WebDashboard:
    """Web-based dashboard for RL training visualization."""
    
    def __init__(self, workload_name: str = "", param_names: list = None, config: dict = None):
        global training_state
        with state_lock:
            training_state['workload_name'] = workload_name
            training_state['param_names'] = param_names or []
            training_state['config'] = config or {}
            training_state['start_time'] = datetime.now()
            training_state['is_training'] = True
    
    def add_data_point(self, step: int, reward: float, performance: float,
                       stability: float, episode: int, params: dict):
        """Add new data point (called from RL training loop)."""
        with state_lock:
            training_state['steps'].append(step)
            training_state['rewards'].append(reward)
            training_state['performance'].append(performance)
            training_state['stability'].append(stability)
            training_state['episodes'].append(episode)
            
            # Track parameter history
            training_state['param_history'].append(params.copy())
            if len(training_state['param_history']) > 100:
                training_state['param_history'].pop(0)
            
            # Update best
            if reward > training_state['best_reward']:
                training_state['best_reward'] = reward
                training_state['best_config'] = params.copy()
                training_state['best_step'] = step
                print(f"[Dashboard] New best! Step {step}, Reward: {reward:.2f}", flush=True)
    
    def stop(self):
        """Mark training as stopped and complete."""
        with state_lock:
            training_state['is_training'] = False
            training_state['training_complete'] = True
    
    def start(self):
        """Mark training as started."""
        with state_lock:
            training_state['is_training'] = True
            training_state['training_complete'] = False


# Flask routes
@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('rl_dashboard.html')


@app.route('/data')
def get_data():
    """Get current training data as JSON."""
    with state_lock:
        # Convert all numpy types to Python native types
        data = {
            'steps': [int(x) for x in training_state['steps']],
            'rewards': [float(x) for x in training_state['rewards']],
            'episodes': [int(x) for x in training_state['episodes']],
            'best_reward': float(training_state['best_reward']) if training_state['best_reward'] is not None else 0.0,
            'best_config': training_state['best_config'],
            'is_training': bool(training_state['is_training']),
            'training_complete': bool(training_state['training_complete']),
            'workload_name': training_state['workload_name'],
            'param_names': training_state['param_names']
        }
        
        # Add config info
        if training_state['config']:
            data['config'] = {
                'reward_metric': training_state['config'].get('reward_metric', 'N/A'),
                'benchmark_command': training_state['config'].get('benchmark_command', 'N/A'),
                'total_timesteps': training_state['config'].get('training_config', {}).get('total_timesteps', 0),
                'learning_rate': training_state['config'].get('training_config', {}).get('learning_rate', 0),
                'action_space': training_state['config'].get('action_space', [])
            }
        
        # Calculate statistics
        if len(training_state['rewards']) > 0:
            data['current_reward'] = float(training_state['rewards'][-1])
            recent = list(training_state['rewards'])[-20:]
            data['avg_reward'] = float(np.mean(recent))
            data['total_steps'] = int(training_state['steps'][-1])
            data['total_episodes'] = int(max(training_state['episodes']))
        else:
            data['current_reward'] = 0.0
            data['avg_reward'] = 0.0
            data['total_steps'] = 0
            data['total_episodes'] = 0
        
        # Calculate runtime
        if training_state['start_time']:
            elapsed = (datetime.now() - training_state['start_time']).total_seconds()
            data['runtime'] = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            data['runtime'] = "0m 0s"
    
    return jsonify(data)


@app.route('/stream')
def stream():
    """Server-Sent Events stream for real-time updates."""
    def event_stream():
        while True:
            with state_lock:
                # Stop streaming if training is complete
                if training_state['training_complete']:
                    yield f"data: {json.dumps({'complete': True})}\n\n"
                    break
                
                if len(training_state['rewards']) > 0:
                    data = {
                        'step': int(training_state['steps'][-1]),
                        'reward': float(training_state['rewards'][-1]),
                        'best_reward': float(training_state['best_reward']),
                        'is_training': training_state['is_training'],
                        'complete': False
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)  # Update every second
    
    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/plot/learning_curve')
def plot_learning_curve():
    """Generate learning curve plot as base64 image."""
    with state_lock:
        if len(training_state['rewards']) == 0:
            return jsonify({'error': 'No data yet'})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = list(training_state['steps'])
        rewards = list(training_state['rewards'])
        
        # Raw rewards
        ax.plot(steps, rewards, alpha=0.3, color='blue', label='Raw Reward', linewidth=1)
        
        # Moving average
        if len(rewards) >= 20:
            window = min(50, len(rewards) // 5)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], moving_avg, color='red', linewidth=2, 
                   label=f'Moving Avg ({window})')
        
        # Mark best
        if training_state['best_step'] in steps:
            idx = steps.index(training_state['best_step'])
            ax.plot(training_state['best_step'], rewards[idx], 'g*', 
                   markersize=15, label=f'Best: {training_state["best_reward"]:.2f}')
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    return jsonify({'image': img_base64})


@app.route('/plot/performance_stability')
def plot_performance_stability():
    """Generate performance vs stability scatter plot."""
    with state_lock:
        if len(training_state['performance']) == 0:
            return jsonify({'error': 'No data yet'})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        perf = list(training_state['performance'])
        stab = list(training_state['stability'])
        steps = list(training_state['steps'])
        
        scatter = ax.scatter(perf, stab, c=steps, cmap='viridis', alpha=0.6, s=30)
        
        ax.set_xlabel('Performance Score', fontsize=12)
        ax.set_ylabel('Stability Score', fontsize=12)
        ax.set_title('Performance vs Stability', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Step')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    return jsonify({'image': img_base64})


@app.route('/plot/episode_rewards')
def plot_episode_rewards():
    """Generate episode rewards bar chart."""
    with state_lock:
        if len(training_state['episodes']) == 0:
            return jsonify({'error': 'No data yet'})
        
        # Group by episode
        episodes = training_state['episodes']
        rewards = training_state['rewards']
        
        unique_episodes = sorted(set(episodes))
        episode_avg_rewards = []
        
        for ep in unique_episodes:
            ep_rewards = [r for e, r in zip(episodes, rewards) if e == ep]
            if ep_rewards:
                episode_avg_rewards.append(np.mean(ep_rewards))
        
        if not episode_avg_rewards:
            return jsonify({'error': 'No episode data yet'})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(unique_episodes, episode_avg_rewards, color='skyblue', alpha=0.7)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    return jsonify({'image': img_base64})


def run_dashboard_server(host='0.0.0.0', port=5000):
    """Run the dashboard web server."""
    # Note: debug=False because this runs in a background thread
    # Debug mode requires signals which only work in main thread
    app.run(host=host, port=port, threaded=True, debug=False)


# For standalone testing
if __name__ == '__main__':
    run_dashboard_server()
