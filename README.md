
HERAgent.py - Implements the HER replay buffer that relabels transitions for improved learning in sparse-reward settings.

SAC_Network_Agent.py - Contains the SAC agent implementation, including both the policy (actor) network and Q-function (critic) networks. This file is fully compatible with the saved model files.

PerformanceAsseser.py - provides evaluation code to test the performance of the trained agent.

Gym_Training_Environment - The main training script. It trains the SAC agent on six different random seeds and saves:

Training metrics to training_results.json

Model checkpoints as policy_model_*.pth.


*prerun.py* - A pre-run test script to quickly verify that your installation and code setup are working correctly before full training.

*final.py* - A script for evaluating a trained model. Edit the indicated model path to choose between pre-trained models or a recently trained model.



installation.md - Contains detailed instructions for installing MuJoCo-related packages and other dependencies.
Note: Please read this file carefully before running any scripts, as the installation of MuJoCo binaries is required for proper functioning.


Install Dependencies:
Follow the instructions in installation.md to install MuJoCo-related packages and set up your Python virtual environment.

