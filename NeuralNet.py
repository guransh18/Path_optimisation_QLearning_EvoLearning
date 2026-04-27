import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# Define the neural network architecture
# class CarNetwork(nn.Module):
#     def __init__(self, input_size=5, hidden_size=16, output_size=4):
#         super(CarNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)  # No activation on output layer for Q-values

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int, dueling: bool = True):
        super().__init__()
        self.dueling = dueling

        self.shared = nn.Sequential(
            nn.Linear(n_inputs, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU()
            # nn.Dropout(0.1),
        )

        if dueling:
            self.value_head     = nn.Linear(64, 1)
            self.advantage_head = nn.Linear(64, n_actions)
        else:
            self.q_head = nn.Linear(64, n_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.shared(x)
        if self.dueling:
            V = self.value_head(features)
            A = self.advantage_head(features)
            return V + A - A.mean(dim=1, keepdim=True)
        return self.q_head(features)

# Define the agent that will use the neural network for Q-learning
# class DQNAgent:
#     def __init__(self, state_size=5, action_size=4, hidden_size=16,
#                  learning_rate=0.001, gamma=0.99, epsilon=1.0, 
#                  epsilon_min=0.5, epsilon_decay=0.995, memory_size=10000, 
#                  batch_size=64):
#         self.state_size = state_size  # Speed, angular speed, 5 ray distances
#         self.action_size = action_size  # W, A, S, D
#         self.memory = deque(maxlen=memory_size)
#         self.gamma = gamma  # Discount factor
#         self.epsilon = epsilon  # Exploration rate
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Main network for training
#         self.model = CarNetwork(state_size, hidden_size, action_size).to(self.device)
#         # Target network for stable learning
#         self.target_model = CarNetwork(state_size, hidden_size, action_size).to(self.device)
#         self.update_target_model()
        
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.criterion = nn.MSELoss()
        
#         # For tracking training progress
#         self.checkpoint_rewards = []
#         self.episode_rewards = []
#         self.best_reward = -float('inf')

class DQNAgent:
    def __init__(self, state_size=7, action_size=8, hidden_size=16,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.5, epsilon_decay=0.9995, memory_size=10000, 
                 batch_size=64):
        self.state_size = state_size  # Speed, angular speed, 5 ray distances
        self.action_size = action_size  # 8 actions
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hook up the new DQN architecture here
        self.model = DQN(n_inputs=state_size, n_actions=action_size).to(self.device)
        self.target_model = DQN(n_inputs=state_size, n_actions=action_size).to(self.device)
        self.update_target_model()
        
        # >>> MAKE SURE THESE TWO LINES ARE ADDED HERE <<<
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For tracking training progress
        self.checkpoint_rewards = []
        self.episode_rewards = []
        self.best_reward = -float('inf')

    def get_weights_flat(self):
        """Return all model weights as a single 1D numpy array.

        The order follows the model's state_dict() iteration order.
        """
        weights = []
        for _, tensor in self.model.state_dict().items():
            weights.append(tensor.detach().cpu().numpy().ravel())
        if len(weights) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(weights).astype(np.float32)

    def set_weights_flat(self, flat_weights):
        """Set model weights from a single 1D array and update target model.

        This will reshape the flat array to match each parameter in the
        model's state_dict(). The model architecture must match.
        """
        flat = np.asarray(flat_weights, dtype=np.float32).ravel()
        state = self.model.state_dict()
        new_state = {}
        idx = 0
        for name, tensor in state.items():
            numel = tensor.numel()
            if idx + numel > flat.size:
                raise ValueError("Not enough weights provided to fill the model.")
            reshaped = flat[idx:idx + numel].reshape(tensor.shape)
            new_state[name] = torch.from_numpy(reshaped).to(tensor.device).type_as(tensor)
            idx += numel

        if idx != flat.size:
            raise ValueError("Too many weights provided for the model.")

        self.model.load_state_dict(new_state)
        self.update_target_model()
        # --- ADD THIS LINE ---
        # Reset the optimizer to clear old momentum that will corrupt the new weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Select action based on current state using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            # Explore: choose random action
            return random.randrange(self.action_size)
        
        # Exploit: bypass BatchNorm crash on single samples
        self.model.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        self.model.train() # Switch back to training mode
        
        return torch.argmax(action_values).item()
    
    def replay(self):
        """Train on batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([experience[0] for experience in minibatch]).to(self.device)
        actions = torch.LongTensor([[experience[1]] for experience in minibatch]).to(self.device)
        rewards = torch.FloatTensor([[experience[2]] for experience in minibatch]).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch]).to(self.device)
        dones = torch.FloatTensor([[experience[4]] for experience in minibatch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions)
        
        # Next Q values from target model (for stability)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1, keepdim=True)[0]
        
        # Target Q values
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss and update weights
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory,
            'checkpoint_rewards': self.checkpoint_rewards,
            'episode_rewards': self.episode_rewards,
            'best_reward': self.best_reward
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = checkpoint.get('memory', self.memory)
            self.checkpoint_rewards = checkpoint.get('checkpoint_rewards', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def get_action_keys(self, action_idx):
        """Convert action index to key press dictionary"""
        # Map actions to key combinations
        # 0: No keys, 1: W, 2: A, 3: S, 4: D, 5: W+A, 6: W+D, 7: S+A, 8: S+D
        keys = {'W': False, 'A': False, 'S': False, 'D': False}
        
        if self.action_size == 4:
            # Simple mapping - one key at a time
            if action_idx == 0:
                keys['W'] = True
            elif action_idx == 1:
                keys['A'] = True
            elif action_idx == 2:
                keys['S'] = True
            elif action_idx == 3:
                keys['D'] = True
        
        elif self.action_size == 8:
            # Complex mapping - combinations allowed
            if action_idx == 0:
                keys['W'] = True
            elif action_idx == 1:
                keys['A'] = True
            elif action_idx == 2:
                keys['S'] = True
            elif action_idx == 3:
                keys['D'] = True
            elif action_idx == 4:
                keys['W'] = True
                keys['A'] = True
            elif action_idx == 5:
                keys['W'] = True
                keys['D'] = True
            elif action_idx == 6:
                keys['S'] = True
                keys['A'] = True
            elif action_idx == 7:
                keys['S'] = True
                keys['D'] = True
        
        return keys

class RewardSystem:
    def __init__(self, initial_points=100, checkpoint_reward=100, decay_rate=0.2):
        self.initial_points = initial_points
        self.remaining_points = initial_points
        self.checkpoint_reward = checkpoint_reward
        self.decay_rate = decay_rate  # Points lost per frame
        self.last_checkpoint_count = 0
        self.total_reward = 0
        
    def calculate_reward(self, car, checkpoints, collision, time_alive):
        """Calculate reward based on decaying points and checkpoints passed"""
        # Handle collision first - set remaining points to zero and return 0
        if collision:
            self.remaining_points = 0  # Immediately set points to zero for termination
            self.total_reward = 0  # No change to total reward on collision
            return 0  # Return exactly 0 as requested for collision
            
        # Normal operation (no collision)
        # Store previous remaining points to calculate reward delta
        previous_points = self.remaining_points
        
        # Decay remaining points over time
        self.remaining_points -= self.decay_rate
        
        # Check if new checkpoints were passed
        current_checkpoint_count = sum(cp.is_passed for cp in checkpoints)
        checkpoints_passed = current_checkpoint_count - self.last_checkpoint_count
        
        if checkpoints_passed > 0:
            # Add checkpoint reward and update remaining points
            checkpoint_bonus = checkpoints_passed * self.checkpoint_reward
            self.remaining_points += checkpoint_bonus
            self.last_checkpoint_count = current_checkpoint_count
        
        # Calculate reward as the change in remaining points
        reward = self.remaining_points - previous_points
        
        # Update total reward
        self.total_reward += reward
        
        return reward
    
    def is_out_of_points(self):
        """Check if the agent has run out of points"""
        return self.remaining_points <= 0
    
    def reset(self):
        """Reset the reward system for a new episode"""
        self.remaining_points = self.initial_points
        self.last_checkpoint_count = 0
        self.total_reward = 0
