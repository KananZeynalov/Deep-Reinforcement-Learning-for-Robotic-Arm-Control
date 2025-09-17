from collections import deque
import random
import numpy as np
import torch

class HindsightReplayBuffer:
    """
    Hindsight Experience Replay (HER) buffer implementation.
    
    HER helps goal-oriented agents learn from failure by relabeling unsuccessful 
    experiences with goals that were actually achieved, turning failures into 
    successful experiences for alternative goals.
    """
    def __init__(self, max_size, strategy="future", num_retrospect=4):
        """
        Initialize the HER replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
            strategy: Strategy for selecting alternative goals ("future" uses achieved states)
            num_retrospect: Number of additional experiences to create through goal relabeling
        """
        self.memory = deque(maxlen=max_size)
        self.relabel_strategy = strategy
        self.num_retrospect = num_retrospect

    def record_experience(self, state, action, reward, next_state, done, achieved, desired):
        """
        Store a transition in the buffer and create additional transitions with HER.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Original reward received
            next_state: Next state observation
            done: Boolean indicating if episode ended
            achieved: Actually achieved goal state
            desired: Original desired goal state
        """
        # Store original experience with the actual goal
        self.memory.append((
            np.array(state), 
            np.array(action),
            reward,
            np.array(next_state),
            done,
            np.array(achieved),
            np.array(desired)
        ))

        # Hindsight goal relabeling: Create additional experiences by pretending
        # the achieved state was actually the goal
        for _ in range(self.num_retrospect):
            # Use the achieved state as a new goal (in hindsight)
            new_target = achieved.copy()
            
            # Recompute reward with respect to the new goal
            # Typically negative distance between achieved and "new" goal
            modified_reward = -np.linalg.norm(achieved - new_target)
            
            # Store the modified experience
            self.memory.append((
                np.array(state),
                np.array(action),
                modified_reward,
                np.array(next_state),
                done,
                np.array(achieved),
                np.array(new_target))
            )

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones, goals)
            ready to be used for neural network training
        """
        # Randomly sample transitions from memory
        transitions = random.sample(self.memory, batch_size)
        
        # Transpose the batch to have all states, actions, etc. grouped together
        batch = list(zip(*transitions))
        
        # Convert to PyTorch tensors and format appropriately
        return (
            torch.FloatTensor(np.array(batch[0])),    # states
            torch.FloatTensor(np.array(batch[1])),    # actions
            torch.FloatTensor(np.array(batch[2])).unsqueeze(-1),  # rewards
            torch.FloatTensor(np.array(batch[3])),    # next_states
            torch.FloatTensor(np.array(batch[4])).unsqueeze(-1),  # dones
            torch.FloatTensor(np.array(batch[6]))     # goals (desired goals)
        )

    def __len__(self):
        """Return the current size of the replay buffer"""
        return len(self.memory)