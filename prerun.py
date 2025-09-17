import gymnasium as gym
import torch
from SAC_Network_Agent import SACAgent
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


# Create the environment (with rendering to visually check behavior)
env = gym.make('FetchReach-v3', render_mode='human')
state, info = env.reset()
# Extract the desired goal from the initial state (this is needed as the networks accept [state, goal])
goal = state.get("desired_goal", None)

# Determine dimensions from the environment's observation and action spaces.
state_dim = state["observation"].shape[0]
action_dim = env.action_space.shape[0]

# Initialize the SAC agent (untrained weights)
agent = SACAgent(state_dim, action_dim)

done = False
while not done:
    # Prepare tensors: note that the network expects state and goal concatenated
    observation = state["observation"]
    state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    goal_tensor = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
    
    # Sample action using the current policy (initially random/untrained)
    action, _ = agent.policy.sample_action(state_tensor, goal_tensor)
    action = action.detach().numpy().squeeze(0)
    
    # Step the environment
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Clean up environment resources
env.close()
print("Pre-training test run finished! Check the rendered display to see the robot's behavior.")
