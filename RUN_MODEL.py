import time
import torch
import gymnasium as gym
from SAC_Network_Agent import SACAgent
from PerformanceAsseser import EnvironmentEvaluator
import numpy as np
import gymnasium_robotics

'''
In this code you can run models I trained with different seeds by correctly adjusting the path.
If you want to train, run Gym_Training_Environment.py, which will take a couple of hours and save model paths to the given destination.
After training you can use too but it wont make much of a difference if you want to save time. Choice is left for the grader. 
'''

# Configuration parameters
EVAL_SEED = 800 # MUST  <-- when you use different paths PLEASE change seed according to that SEED. 
MODEL_PATH = "trained_models/policy_model_800.pth" # if you train model yourself please adjust the path accordingly
NUM_EVAL_EPISODES = 10

def main():
    # Register custom environments
    gym.register_envs(gymnasium_robotics)

    # Environment setup for dimension detection
    temp_env = gym.make('FetchReach-v3')
    init_state, _ = temp_env.reset(seed=EVAL_SEED)
    state_size = init_state["observation"].shape[0]
    action_size = temp_env.action_space.shape[0]
    temp_env.close()

    # Initialize learning agent with trained policy
    policy_agent = SACAgent(state_size, action_size)
    policy_agent.load_model(MODEL_PATH)

    # Configure performance evaluator with visualization
    performance_tester = EnvironmentEvaluator(
        "FetchReach-v3", 
        eval_episodes=NUM_EVAL_EPISODES,
        render_env=True
    )
    performance_tester.setup_simulation(EVAL_SEED)

    # Monkey-patch the step method to add a delay after each step.
    # This will slow down the progress so we can better see the simulation.
    original_step = performance_tester.sim_env.step
    def delayed_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        time.sleep(0.1)  # Adjust delay (in seconds) as needed
        return result
    performance_tester.sim_env.step = delayed_step

    # Conduct evaluation
    success_percentage, mean_return = performance_tester.run_evaluation(
        policy_agent, EVAL_SEED
    )
    
    # Display results
    print("\n" + "="*40)
    print(f"Evaluation Seed: {EVAL_SEED}")
    print(f"Model Path: {MODEL_PATH}")
    print("-"*40)
    print(f"Success Percentage: {success_percentage:.2f}%")
    print(f"Average Episode Return: {mean_return:.2f}")
    print("="*40)

    # Cleanup resources
    performance_tester.terminate()

if __name__ == "__main__":
    main()
