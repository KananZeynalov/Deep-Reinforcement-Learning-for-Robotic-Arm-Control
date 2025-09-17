import json
import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from SAC_Network_Agent import SACAgent
from HERAgent import HindsightReplayBuffer
from PerformanceAsseser import EnvironmentEvaluator

class TrainingOrchestrator:
    """
    Orchestrates the training process for a robotic agent using Soft Actor-Critic (SAC)
    with Hindsight Experience Replay (HER) in the FetchReach environment.
    """
    def __init__(self, state_size, action_size, 
                 memory_capacity=10000, batch_size=64, seed=42):
        """
        Initialize training environment, agent and memory buffer.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            memory_capacity: Capacity of the replay buffer
            batch_size: Batch size for training updates
            seed: Random seed for reproducibility
        """
        gym.register_envs(gymnasium_robotics)
        self.seed = seed
        self.configure_seeding()
        
        # Initialize environment and evaluation utilities
        self.evaluator = EnvironmentEvaluator("FetchReach-v3")
        self.sim_env = gym.make('FetchReach-v3', render_mode=None, 
                              max_episode_steps=100)
        self.sim_env.reset(seed=seed)
        
        # Initialize agent and replay buffer
        self.learning_agent = SACAgent(state_size, action_size)
        self.memory_buffer = HindsightReplayBuffer(memory_capacity)
        self.batch_size = batch_size
        
        # Metrics tracking
        self.training_metrics = {
            "episode_rewards": [],
            "q_losses": [[], []],
            "policy_losses": [],
            "alpha_losses": [],
            "alpha_values": [],
            "goal_distances": [],
            "success_rates": [],
            "eval_rewards": []
        }
        
        # Create directory for saving models
        self.model_dir = "trained_models_grader/"
        os.makedirs(self.model_dir, exist_ok=True)

    def configure_seeding(self):
        """Set random seeds for reproducibility across all libraries"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        gym.utils.seeding.np_random(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def execute_training(self, total_episodes=500, updates_per_episode=100):
        """
        Execute the main training loop for the robotic agent.
        
        Args:
            total_episodes: Total number of training episodes
            updates_per_episode: Number of network updates per episode
            
        Returns:
            Dictionary containing training metrics and logs
        """
        training_log = {
            "total_rewards": [],
            "q_losses": [[], []],
            "policy_losses": [],
            "alpha_losses": [],
            "alpha_values": [],
            "goal_distances": [],
            "success_rates": [],
            "eval_rewards": []
        }

        for episode in tqdm(range(total_episodes), desc=f"Training (seed={self.seed})"):
            # Reset environment at the start of each episode
            env_state, info = self.sim_env.reset()
            target_goal = env_state.get("desired_goal")
            episode_reward = 0
            episode_complete = False

            # Episode interaction loop
            while not episode_complete:
                # Get current state and prepare tensors
                current_obs = env_state["observation"]
                obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
                goal_tensor = torch.FloatTensor(target_goal).unsqueeze(0)

                # Sample action from policy
                action, _ = self.learning_agent.policy.sample_action(obs_tensor, goal_tensor)
                numpy_action = action.detach().numpy().squeeze(0)

                # Take action in environment
                next_state, reward, terminated, truncated, info = \
                    self.sim_env.step(numpy_action)
                
                # Track distance to goal for monitoring progress
                achieved = env_state.get("achieved_goal")
                desired = env_state.get("desired_goal")
                training_log["goal_distances"].append(
                    np.linalg.norm(achieved - desired)
                )

                # Store experience in replay buffer with HER
                self.memory_buffer.record_experience(
                    current_obs, numpy_action, reward, 
                    next_state["observation"], terminated or truncated, 
                    achieved, desired
                )

                # Update state and check if episode is complete
                episode_reward += reward
                env_state = next_state
                episode_complete = terminated or truncated

            # Record episode reward
            training_log["total_rewards"].append(float(episode_reward))

            # Update networks multiple times after each episode
            for _ in range(updates_per_episode):
                if len(self.memory_buffer) > self.batch_size:
                    # Sample batch and update agent
                    batch_data = self.memory_buffer.sample_batch(self.batch_size)
                    losses = self.learning_agent.train(batch_data)
                    
                    # Record training metrics
                    training_log["q_losses"][0].append(losses[0])
                    training_log["q_losses"][1].append(losses[1])
                    training_log["policy_losses"].append(losses[2])
                    training_log["alpha_losses"].append(losses[3])
                    training_log["alpha_values"].append(losses[4].item())

            # Periodic evaluation to track performance
            if episode % 20 == 0 and episode > 0:
                success_rate, avg_reward = self.evaluator.run_evaluation(
                    self.learning_agent, self.seed
                )
                training_log["success_rates"].append({episode: success_rate})
                training_log["eval_rewards"].append({episode: avg_reward})

        # Save the trained model
        model_path = f"{self.model_dir}policy_model_{self.seed}.pth"
        self.learning_agent.save_model(model_path)
        return training_log

if __name__ == "__main__":
    # Run experiments with different seeds
    seed_values = [32, 78, 10030, 11, 812, 5]  # changed seeds for new training you can adjust in this range for each value
    experiment_results = {}

    for seed in seed_values:
        print(f"Initializing training with seed {seed}")
        # Create environment to get dimensions
        env = gym.make('FetchReach-v3')
        state_dim = env.observation_space['observation'].shape[0]
        action_dim = env.action_space.shape[0]
        
        # Initialize trainer and run training
        trainer = TrainingOrchestrator(state_dim, action_dim, 
                                     seed=seed, batch_size=400)
        training_log = trainer.execute_training(500, 100)
        experiment_results[seed] = training_log
        
    # Save results to JSON file
    with open("training_results.json", "w") as f:
        json.dump(experiment_results, f, indent=4, default=lambda o: o.item() if isinstance(o, np.generic) else o)