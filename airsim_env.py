import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
from gym import spaces
import airsim
import cv2
import time
from collections import deque

# SAC implementation for AirSim car control
class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(SACPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std for the Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Sample from the distribution
        x_t = normal.rsample()
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        
        # Calculate log probability, adding correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2


class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, device='cuda'):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        
        # Initialize policy
        self.policy = SACPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize Q networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target Q networks (initialized with same weights)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)
            
        # Initialize automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            # Deterministic action for evaluation
            with torch.no_grad():
                action = self.policy.deterministic_action(state)
        else:
            # Sample action for training
            with torch.no_grad():
                action, _, _ = self.policy.sample(state)
                
        return action.cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size):
        # Sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # Sample next action and its log probability
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            
            # Compute target Q values
            target_q1, target_q2 = self.target_q_network(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        # Compute current Q values
        current_q1, current_q2 = self.q_network(state_batch, action_batch)
        
        # Compute Q network loss
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update Q networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Compute policy loss
        action_new, log_prob, _ = self.policy.sample(state_batch)
        q1_new, q2_new = self.q_network(state_batch, action_new)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return q_loss.item(), policy_loss.item(), alpha_loss.item()
    
    def save(self, directory):
        torch.save(self.policy.state_dict(), os.path.join(directory, 'policy.pth'))
        torch.save(self.q_network.state_dict(), os.path.join(directory, 'q_network.pth'))
        torch.save(self.target_q_network.state_dict(), os.path.join(directory, 'target_q_network.pth'))
        
    def load(self, directory):
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth')))
        self.q_network.load_state_dict(torch.load(os.path.join(directory, 'q_network.pth')))
        self.target_q_network.load_state_dict(torch.load(os.path.join(directory, 'target_q_network.pth')))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)


class AirSimCarEnv:
    def __init__(self, image_shape=(240, 320, 3), seq_length=10):
        # Connect to AirSim
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
        # Reset car
        self.client.reset()
        
        # Camera names
        self.camera_names = ['center_camera', 'left_camera', 'right_camera']
        
        # Image shape
        self.image_shape = image_shape
        
        # Sequence length for temporal data
        self.seq_length = seq_length
        
        # Initialize observation and action spaces
        # State: Images from 3 cameras + car state (speed, previous steering, previous throttle)
        self.observation_space = spaces.Dict({
            'images': spaces.Box(low=0, high=255, shape=(3, *image_shape), dtype=np.uint8),
            'car_state': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        # Action: [steering, throttle]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Initialize history buffers for temporal data
        self.image_history = deque(maxlen=seq_length)
        self.state_history = deque(maxlen=seq_length)
        
        # Initialize previous action
        self.prev_action = np.array([0.0, 0.0])  # [steering, throttle]
        
        # Initialize collision detector
        self.collision_history = []
        
        # Initialize episode info
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        # Initialize reward parameters
        self.speed_weight = 0.1
        self.collision_penalty = -100.0
        self.lane_deviation_penalty = -1.0
        
        print("AirSimCarEnv initialized.")
    
    def _get_observation(self):
        # Get car state
        car_state = self.client.getCarState()
        
        # Extract speed
        speed = car_state.speed
        
        # Create car state vector [speed, previous steering, previous throttle]
        car_state_vector = np.array([speed, self.prev_action[0], self.prev_action[1]], dtype=np.float32)
        
        # Get images from all cameras
        images = []
        for camera_name in self.camera_names:
            response = self.client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            images.append(img_rgb)
        
        # Stack images
        images = np.array(images)
        
        # Update history
        self.image_history.append(images)
        self.state_history.append(car_state_vector)
        
        # If history is not full yet, duplicate the current observation
        while len(self.image_history) < self.seq_length:
            self.image_history.append(images)
            self.state_history.append(car_state_vector)
        
        return {
            'images': images,
            'car_state': car_state_vector,
            'image_history': np.array(list(self.image_history)),
            'state_history': np.array(list(self.state_history))
        }
    
    def _compute_reward(self, action):
        # Get car state
        car_state = self.client.getCarState()
        
        # Extract speed
        speed = car_state.speed
        
        # Check for collision
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided
        
        if has_collided:
            self.collision_history.append(True)
            return self.collision_penalty, True
        
        # Reward for speed
        speed_reward = self.speed_weight * speed
        
        # Penalty for lane deviation (simplified, would need lane detection in real implementation)
        # Here we use center camera to estimate if car is on the road
        center_img = self._get_observation()['images'][0]  # Center camera
        
        # Simple lane detection (assuming road is darker than surroundings)
        # This is a placeholder - real implementation would need proper lane detection
        gray = cv2.cvtColor(center_img, cv2.COLOR_RGB2GRAY)
        road_mask = gray < 100  # Assuming road is dark
        road_percentage = np.mean(road_mask)
        
        if road_percentage < 0.3:  # If less than 30% of the image is road
            lane_penalty = self.lane_deviation_penalty
        else:
            lane_penalty = 0.0
        
        # Combine rewards
        reward = speed_reward + lane_penalty
        
        # Check if episode should end
        done = False
        self.episode_step += 1
        if self.episode_step >= self.max_episode_steps:
            done = True
        
        return reward, done
    
    def reset(self):
        # Reset car
        self.client.reset()
        self.client.enableApiControl(True)
        
        # Reset episode info
        self.episode_step = 0
        self.collision_history = []
        
        # Reset action
        self.prev_action = np.array([0.0, 0.0])
        
        # Reset history buffers
        self.image_history.clear()
        self.state_history.clear()
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action):
        # Apply action
        controls = airsim.CarControls()
        controls.steering = float(action[0])  # -1 to 1
        controls.throttle = float(action[1])  # 0 to 1
        controls.brake = 0
        
        self.client.setCarControls(controls)
        
        # Save action for next observation
        self.prev_action = action
        
        # Wait for physics to settle
        time.sleep(0.1)
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward and done flag
        reward, done = self._compute_reward(action)
        
        # Additional info
        info = {
            'collision': len(self.collision_history) > 0,
            'episode_step': self.episode_step
        }
        
        return obs, reward, done, info
    
    def close(self):
        # Return control to user
        self.client.enableApiControl(False)
        print("Control returned to user. AirSimCarEnv closed.")
