import gymnasium_robotics
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
from SAC_Network_Agent import SACAgent

class EnvironmentEvaluator:
    """
    Evaluates a trained agent's performance in robotics environments.
    
    This class handles setting up the environment, running multiple evaluation episodes,
    and calculating metrics like success rate and average reward.
    """
    def __init__(self, environment_name, eval_episodes=100, render_env=False):
        """
        Initialize the evaluator with environment settings.
        
        Args:
            environment_name: Name of the Gym environment to evaluate in
            eval_episodes: Number of episodes to run for evaluation
            render_env: Whether to render the environment visually
        """
        self.environment_name = environment_name
        self.eval_episodes = eval_episodes
        self.render_env = render_env
        self.sim_env = None

    def setup_simulation(self, seed_val=None):
        """
        Create and configure the simulation environment.
        
        Args:
            seed_val: Random seed for reproducibility
        """
        # Set rendering mode based on configuration
        render_mode = "human" if self.render_env else None
        self.sim_env = gym.make(self.environment_name, render_mode=render_mode)
        
        # Set random seed if provided
        if seed_val is not None:
            gym.utils.seeding.np_random(seed_val)

    def run_evaluation(self, agent, seed_val=None):
        """
        Run multiple evaluation episodes and calculate performance metrics.
        
        Args:
            agent: The trained agent to evaluate
            seed_val: Random seed for reproducibility
            
        Returns:
            Tuple of (success_rate, average_reward)
        """
        # Create environment if not already created
        if self.sim_env is None:
            self.setup_simulation(seed_val)

        # Metrics tracking
        success_counter = 0
        reward_tracker = []

        # Disable gradient computation during evaluation
        with torch.no_grad():
            # Progress tracking for multiple episodes
            progress_bar = tqdm(range(self.eval_episodes), 
                              desc=f"Evaluation (seed={seed_val})")
            
            for _ in progress_bar:
                # Reset environment for new episode
                env_state, info = self.sim_env.reset()
                target_goal = env_state.get("desired_goal")
                episode_complete = False
                total_episode_reward = 0

                # Run single episode until completion
                while not episode_complete:
                    # Prepare observation for agent
                    current_obs = env_state["observation"]
                    obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
                    goal_tensor = torch.FloatTensor(target_goal).unsqueeze(0)

                    # Get action from policy
                    action, _ = agent.policy.sample_action(obs_tensor, goal_tensor)
                    numpy_action = action.cpu().numpy().squeeze(0)

                    # Execute action in environment
                    next_state, reward, terminated, truncated, info = \
                        self.sim_env.step(numpy_action)
                    
                    # Check for episode completion
                    episode_complete = terminated or truncated
                    total_episode_reward += reward

                    # In FetchReach, reward of 0 indicates success
                    if reward == 0:
                        success_counter += 1
                        break

                    # Update state
                    env_state = next_state

                # Record episode statistics
                reward_tracker.append(total_episode_reward)

        # Calculate overall metrics
        success_rate = (success_counter / self.eval_episodes) * 100
        avg_reward = np.mean(reward_tracker)
        return success_rate, avg_reward

    def terminate(self):
        """Clean up environment resources"""
        if self.sim_env:
            self.sim_env.close()


if __name__ == "__main__":
    # Register robotics environments
    gym.register_envs(gymnasium_robotics)
    
    # Configuration
    seeds = [32, 78, 10030, 11, 812, 5]  # changed seeds for new training you can adjust in this range for each value
    # Create temporary environment to get dimensions
    temp_env = gym.make('FetchReach-v3')
    state_dim = temp_env.observation_space['observation'].shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    # Initialize agent and load pre-trained model
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    model_path = "saved_model_42.pth"  # Path to the saved model
    agent.load_model(model_path)
    
    # Run evaluation with different seeds
    for seed in seeds:
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize evaluator and run evaluation
        evaluator = EnvironmentEvaluator("FetchReach-v3")
        evaluator.setup_simulation(seed)
        success_rate, avg_reward = evaluator.run_evaluation(agent, seed)
        
        # Print results
        print(f"Seed {seed}: Success rate: {success_rate:.2f}%, avg_reward: {avg_reward:.2f}")
        
        # Clean up
        evaluator.terminate()
        del evaluator