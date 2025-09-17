import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from HERAgent import HindsightReplayBuffer

class PolicyNN(nn.Module):
    """
    Policy Network (Actor) that maps states and goals to a distribution over actions.
    Uses a Gaussian distribution with tanh squashing to bound actions.
    """
    def __init__(self, state_dimension, action_dimension):
        super(PolicyNN, self).__init__()
        # Network architecture
        self.fully_connected1 = nn.Linear(state_dimension + 3, 256)
        self.fully_connected2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dimension)  # Output mean of action distribution
        self.log_std = nn.Linear(256, action_dimension)  # Output log std of action distribution
        
        # Action scaling parameters
        self.action_scale = torch.tensor(1.0, dtype=torch.float32)
        self.action_bias = torch.tensor(0.0, dtype=torch.float32)
    
    def forward(self, state, goal):
        """
        Forward pass to compute mean and log_std of action distribution.
        
        Args:
            state: Current state tensor
            goal: Goal state tensor
            
        Returns:
            mean: Mean of the Gaussian distribution
            log_std: Log standard deviation of the Gaussian distribution
        """
        # Concatenate state and goal as input
        x = torch.cat([state, goal], dim=1) 
        x = torch.relu(self.fully_connected1(x))
        x = torch.relu(self.fully_connected2(x))
        
        # Output mean and log_std
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Clamp for numerical stability
        return mean, log_std
    
    def sample_action(self, state, goal):
        """
        Sample actions from the policy using the reparameterization trick.
        
        Args:
            state: Current state tensor
            goal: Goal state tensor
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the sampled action
        """
        # Get distribution parameters
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()

        # Sample using reparameterization trick
        normal = dist.Normal(mean, std)
        u = normal.rsample()  # Reparameterized sample
        y_t = torch.tanh(u)  # Apply tanh squashing
        
        # Scale and shift action to desired range
        action = y_t * self.action_scale + self.action_bias

        # Calculate log probability, accounting for tanh transformation
        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # Sum over all dimensions

        return action, log_prob

class QFuncNN(nn.Module):
    """
    Q-Function Network (Critic) that estimates the Q-value for state-action pairs.
    Uses double Q-networks to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim):
        super(QFuncNN, self).__init__()
        
        # First Q-network
        self.fully_connected1 = nn.Linear(state_dim + 3 + action_dim, 256)
        self.fully_connected2 = nn.Linear(256, 256)
        self.out1 = nn.Linear(256, 1)

        # Second Q-network (for reducing overestimation bias)
        self.fully_connected3 = nn.Linear(state_dim + 3 + action_dim, 256)
        self.fully_connected4 = nn.Linear(256, 256)
        self.out2 = nn.Linear(256, 1)
    
    def forward(self, state, action, goal):
        """
        Forward pass to compute Q-values.
        
        Args:
            state: Current state tensor
            action: Action tensor
            goal: Goal state tensor
            
        Returns:
            q1, q2: Q-values from both networks
        """
        # Concatenate state, goal, and action
        x_input = torch.cat([state, goal, action], dim=-1)
        
        # First Q-network forward pass
        x1 = F.relu(self.fully_connected1(x_input))
        x1 = F.relu(self.fully_connected2(x1))
        x1 = self.out1(x1)

        # Second Q-network forward pass
        x2 = F.relu(self.fully_connected3(x_input))
        x2 = F.relu(self.fully_connected4(x2))
        x2 = self.out2(x2)
        
        return x1, x2

class SACAgent:
    """
    Soft Actor-Critic (SAC) agent that combines policy and Q-function networks
    with entropy regularization for improved exploration.
    """
    def __init__(self, state_dim, action_dim, target_entropy=-2.0):
        """
        Initialize SAC agent with policy and Q networks.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            target_entropy: Target entropy for automatic temperature adjustment
        """
        # Initialize networks
        self.policy = PolicyNN(state_dim, action_dim)
        self.QFunction = QFuncNN(state_dim, action_dim)
        self.target_QFunction = QFuncNN(state_dim, action_dim)

        # Copy parameters from Q-function to target Q-function
        self.target_QFunction.load_state_dict(self.QFunction.state_dict())

        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(self.QFunction.parameters(), lr=3e-4)
        
        # Temperature parameter for entropy regularization
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = target_entropy

    def train(self, batch, gamma=0.99, tau=0.005):
        """
        Main training method that updates all components of the agent.
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones, goals)
            gamma: Discount factor
            tau: Soft update coefficient
            
        Returns:
            Tuple of loss values and current alpha
        """
        # Update critic (Q-functions)
        q1_loss, q2_loss = self.update_q_functions(batch, gamma)

        # Update actor (policy)
        policy_loss, log_probs = self.update_policy(batch)

        # Update temperature parameter
        alpha_loss = self.update_alpha(log_probs)

        # Soft update target networks
        self.soft_update(self.target_QFunction, self.QFunction, tau)
        
        return q1_loss, q2_loss, policy_loss, alpha_loss, self.alpha

    def update_policy(self, batch):
        """
        Update policy network to maximize expected return and entropy.
        
        Args:
            batch: Experience batch from replay buffer
            
        Returns:
            policy_loss: Loss value for the policy update
            log_probs: Log probabilities of sampled actions
        """
        states, _, _, _, _, new_goal = batch
        
        # Sample actions and get log probabilities
        actions, log_probs = self.policy.sample_action(states, new_goal)

        # Evaluate actions with Q-function
        q1, q2 = self.QFunction(states, actions, new_goal)
        q_values = torch.min(q1, q2)  # Use minimum Q-value for stability
        
        # Policy loss: maximize Q-value and entropy (negative because we're minimizing)
        policy_loss = (self.alpha * log_probs - q_values).mean()

        # Optimize the policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item(), log_probs

    def update_q_functions(self, batch, gamma=0.99):
        """
        Update Q-function networks using TD learning.
        
        Args:
            batch: Experience batch from replay buffer
            gamma: Discount factor
            
        Returns:
            q1_loss, q2_loss: Loss values for both Q-networks
        """
        states, actions, rewards, next_states, dones, new_goal = batch
        
        # Calculate target Q-values (without gradient)
        with torch.no_grad():
            # Get next actions from current policy
            next_actions, next_log_probs = self.policy.sample_action(next_states, new_goal)
            
            # Get target Q-values
            target_q1, target_q2 = self.target_QFunction(next_states, next_actions, new_goal)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + gamma * (1 - dones) * target_q

        # Current Q-value estimates
        current_q1, current_q2 = self.QFunction(states, actions, new_goal)

        # Compute TD errors
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        # Update Q-functions
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q1_loss.item(), q2_loss.item()

    def update_alpha(self, log_probs):
        """
        Update temperature parameter alpha to maintain target entropy.
        
        Args:
            log_probs: Log probabilities of actions from current policy
            
        Returns:
            alpha_loss: Loss value for alpha update
        """
        # Alpha loss: try to keep policy entropy close to target entropy
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update actual alpha value from log_alpha
        self.alpha = self.log_alpha.exp()

        return alpha_loss.item()

    def soft_update(self, target, source, tau):
        """
        Soft update target network parameters:
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            target: Target network
            source: Source network
            tau: Update coefficient (small value for slow update)
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, save_path="sac_model.pth"):
        """
        Save model parameters to disk.
        
        Args:
            save_path: Path to save model
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'QFunction_state_dict': self.QFunction.state_dict(),
            'target_QFunction_state_dict': self.target_QFunction.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path="sac_model.pth"):
        """
        Load model parameters from disk.
        
        Args:
            load_path: Path to load model from
        """
        checkpoint = torch.load(load_path, weights_only=True)
        
        # Load network parameters
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.QFunction.load_state_dict(checkpoint['QFunction_state_dict'])
        self.target_QFunction.load_state_dict(checkpoint['target_QFunction_state_dict'])
        
        # Load optimizer states and temperature parameter
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        
        # Update alpha value
        self.alpha = self.log_alpha.exp()

        print(f"Model loaded from {load_path}")