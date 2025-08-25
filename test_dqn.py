#!/usr/bin/env python3
"""
Quick test script for DQN implementation.

This script verifies that the DQN components work correctly before
running full training experiments for the IEEE paper comparison.
"""

import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperparameters import Config
from agents.dqn_agent import DQNAgent
from environment.dqn_environment_wrapper import DQNEnvironmentWrapper
from utilities.action_discretizer import ActionDiscretizer


def test_action_discretizer():
    """Test action discretization utilities."""
    print("Testing Action Discretizer...")
    
    config = Config()
    action_bounds = {
        'hvac_energy': config.get('environment', 'hvac_action_bounds'),
        'battery_action': config.get('environment', 'battery_action_bounds'),
        'selling_price': config.get('environment', 'selling_price_bounds')
    }
    
    discretizer = ActionDiscretizer(action_bounds)
    
    # Test discrete to continuous conversion
    discrete_action = 1050  # Middle of action space
    continuous_action = discretizer.discrete_to_continuous(discrete_action)
    print(f"Discrete action {discrete_action} -> Continuous: {continuous_action}")
    
    # Test continuous to discrete conversion
    back_to_discrete = discretizer.continuous_to_discrete(continuous_action)
    print(f"Continuous {continuous_action} -> Discrete: {back_to_discrete}")
    
    # Test action space info
    info = discretizer.get_action_space_info()
    print(f"Action space info: {info}")
    
    # Test random sampling
    random_action = discretizer.sample_random_action()
    random_continuous = discretizer.discrete_to_continuous(random_action)
    print(f"Random discrete action {random_action} -> {random_continuous}")
    
    print("✓ Action Discretizer tests passed!\n")


def test_dqn_environment():
    """Test DQN environment wrapper."""
    print("Testing DQN Environment Wrapper...")
    
    env = DQNEnvironmentWrapper(dynamic=False, eval_mode=False)
    
    # Test environment reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State dimension: {env.total_state_dim}")
    
    # Test action space info
    action_info = env.get_action_space_info()
    print(f"Action space info: {action_info}")
    
    # Test environment step - FIXED: Generate independent actions per house
    house_actions = [env.sample_random_discrete_action() for _ in range(env.num_houses)]
    
    next_state, reward, done, info = env.step(house_actions)
    print(f"Step result: state_shape={next_state.shape}, reward={reward:.3f}, done={done}")
    print(f"Action info: {info.get('discrete_actions', [])}")
    
    # Test a few more steps - FIXED: Generate independent actions per house
    for i in range(3):
        house_actions = [env.sample_random_discrete_action() for _ in range(env.num_houses)]
        next_state, reward, done, info = env.step(house_actions)
        print(f"Step {i+2}: reward={reward:.3f}, done={done}")
        if done:
            break
    
    # Test action usage stats
    usage_stats = env.get_action_usage_stats()
    print(f"Action usage stats: {list(usage_stats.keys())}")
    
    print("✓ DQN Environment Wrapper tests passed!\n")


def test_dqn_agent():
    """Test DQN agent functionality."""
    print("Testing DQN Agent...")
    
    config = Config()
    
    # Get action bounds
    action_bounds = {
        'hvac_energy': config.get('environment', 'hvac_action_bounds'),
        'battery_action': config.get('environment', 'battery_action_bounds'),
        'selling_price': config.get('environment', 'selling_price_bounds')
    }
    
    # Create agent
    state_dim = 170  # Based on environment configuration
    agent = DQNAgent(
        state_dim=state_dim,
        action_bounds=action_bounds,
        config=config
    )
    
    print(f"Agent created with {agent.num_actions} discrete actions")
    
    # Test action selection
    dummy_state = np.random.randn(state_dim)
    
    # Test exploration action
    action_explore = agent.select_action(dummy_state, evaluation=False)
    print(f"Exploration action: {action_explore}")
    
    # Test greedy action
    action_greedy = agent.select_action(dummy_state, evaluation=True)
    print(f"Greedy action: {action_greedy}")
    
    # Test continuous action conversion
    continuous_action = agent.get_continuous_action(dummy_state, evaluation=True)
    print(f"Continuous action: {continuous_action}")
    
    # Test experience storage and learning
    next_state = np.random.randn(state_dim)
    reward = 10.5
    done = False
    
    agent.store_transition(dummy_state, action_explore, reward, next_state, done)
    print("✓ Experience stored")
    
    # Store a few more experiences
    for _ in range(100):
        s = np.random.randn(state_dim)
        a = np.random.randint(0, agent.num_actions)
        r = np.random.randn()
        ns = np.random.randn(state_dim)
        d = np.random.rand() < 0.1
        agent.store_transition(s, a, r, ns, d)
    
    # Test learning
    loss = agent.learn()
    print(f"Learning step loss: {loss}")
    
    # Test statistics
    stats = agent.get_statistics()
    print(f"Agent statistics: {stats}")
    
    print("✓ DQN Agent tests passed!\n")


def test_integration():
    """Test integration of all components."""
    print("Testing DQN Integration...")
    
    config = Config()
    
    # Create environment and agent
    env = DQNEnvironmentWrapper(dynamic=False, eval_mode=False)
    agent = DQNAgent(
        state_dim=env.total_state_dim,
        action_bounds={
            'hvac_energy': config.get('environment', 'hvac_action_bounds'),
            'battery_action': config.get('environment', 'battery_action_bounds'),
            'selling_price': config.get('environment', 'selling_price_bounds')
        },
        config=config
    )
    
    # Run a short episode
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(20):  # Short episode
        # FIXED: Generate independent actions per house instead of broadcasting
        house_actions = []
        for house_idx in range(env.num_houses):
            discrete_action = agent.select_action(state, evaluation=False)
            house_actions.append(discrete_action)
        
        # Use first action for experience storage (representative)
        representative_action = house_actions[0]
        
        # Take step
        next_state, reward, done, info = env.step(house_actions)
        
        # Store experience using representative action
        agent.store_transition(state, representative_action, reward, next_state, done)
        
        # Learn (after some experience)
        if step > 10:
            loss = agent.learn()
        
        # Update
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"Integration test completed:")
    print(f"  - Steps: {steps}")
    print(f"  - Total reward: {total_reward:.3f}")
    print(f"  - Final epsilon: {agent.epsilon:.3f}")
    print(f"  - Memory size: {len(agent.memory) if hasattr(agent.memory, '__len__') else agent.memory.size}")
    
    print("✓ Integration test passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("DQN IMPLEMENTATION VERIFICATION TESTS")
    print("="*60)
    print()
    
    try:
        test_action_discretizer()
        test_dqn_environment()
        test_dqn_agent()
        test_integration()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED!")
        print("DQN implementation is ready for training and comparison with DDPG.")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())