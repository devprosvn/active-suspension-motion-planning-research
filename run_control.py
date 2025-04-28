import os
import numpy as np
import torch
import airsim
import time
import cv2
from collections import deque
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.mobilevit import VisionTemporalModel
from airsim_env import SAC, AirSimCarEnv, ReplayBuffer

# Custom dataset for driving data
class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        if not os.path.exists(csv_file):
            print(f"CSV file not found at: {csv_file}")
            self.data_frame = pd.DataFrame()
        else:
            self.data_frame = pd.read_csv(csv_file)
            print(f"CSV file loaded from: {csv_file}, found {len(self.data_frame)} samples")
        
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(root_dir):
            print(f"Root directory not found at: {root_dir}")
        
        # Filter out invalid image paths
        valid_rows = []
        for idx in range(len(self.data_frame)):
            row = self.data_frame.iloc[idx]
            # First, try the direct path
            center_path = os.path.join(self.root_dir, row['center_image'])
            left_path = os.path.join(self.root_dir, row['left_image'])
            right_path = os.path.join(self.root_dir, row['right_image'])
            
            # If not found, try under 'images' subdirectory
            if not (os.path.exists(center_path) and os.path.exists(left_path) and os.path.exists(right_path)):
                center_path = os.path.join(self.root_dir, 'images', row['center_image'])
                left_path = os.path.join(self.root_dir, 'images', row['left_image'])
                right_path = os.path.join(self.root_dir, 'images', row['right_image'])
                
            if os.path.exists(center_path) and os.path.exists(left_path) and os.path.exists(right_path):
                valid_rows.append(idx)
            else:
                print(f"Skipping sample {idx}: Images not found at {center_path}, {left_path}, or {right_path}")
        self.data_frame = self.data_frame.iloc[valid_rows].reset_index(drop=True)
        print(f"Loaded {len(self.data_frame)} valid data samples")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load images
        row = self.data_frame.iloc[idx]
        # First, try the direct path
        center_path = os.path.join(self.root_dir, row['center_image'])
        left_path = os.path.join(self.root_dir, row['left_image'])
        right_path = os.path.join(self.root_dir, row['right_image'])
        
        # If not found, try under 'images' subdirectory
        if not (os.path.exists(center_path) and os.path.exists(left_path) and os.path.exists(right_path)):
            center_path = os.path.join(self.root_dir, 'images', row['center_image'])
            left_path = os.path.join(self.root_dir, 'images', row['left_image'])
            right_path = os.path.join(self.root_dir, 'images', row['right_image'])
        
        center_img = cv2.imread(center_path)
        # No need to load left and right images since we're only using center
        if center_img is None:
            raise ValueError(f"Failed to load center image at index {idx} from path {center_path}")
        
        center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
        
        # Apply transform to center image only
        if self.transform:
            center_img = self.transform(center_img)
        
        # Use only center image with sequence dimension
        images = center_img.unsqueeze(0)  # Shape: (1, C, H, W), where C=3 for RGB
        
        # Get steering and throttle
        steering = float(row['steering'])
        throttle = float(row['throttle'])
        labels = torch.tensor([steering, throttle], dtype=torch.float32)
        
        return images, labels

def train_sac(env, agent, replay_buffer, num_episodes=1000, batch_size=64, updates_per_step=1, 
              save_interval=100, model_dir='saved_models', eval_interval=10):
    """
    Train the SAC agent in the AirSim environment
    """
    # Create directory for saving models
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize variables
    total_steps = 0
    episode_rewards = []
    
    print(f"Starting SAC training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        episode_steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(np.concatenate([
                state['image_history'].reshape(-1),
                state['state_history'].reshape(-1)
            ]))
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(
                np.concatenate([state['image_history'].reshape(-1), state['state_history'].reshape(-1)]),
                action,
                reward,
                np.concatenate([next_state['image_history'].reshape(-1), next_state['state_history'].reshape(-1)]),
                float(done)
            )
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Update agent
            if len(replay_buffer) > batch_size:
                for _ in range(updates_per_step):
                    q_loss, policy_loss, alpha_loss = agent.update_parameters(replay_buffer, batch_size)
            
            # Check for early termination
            if done:
                break
        
        # Log episode results
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, Avg Reward = {avg_reward:.2f}")
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(model_dir)
            print(f"Model saved to {model_dir} at episode {episode}")
        
        # Evaluate agent periodically
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, num_episodes=3)
            print(f"Evaluation at episode {episode}: Avg Reward = {eval_reward:.2f}")
    
    # Save final model
    agent.save(model_dir)
    print(f"Training completed. Final model saved to {model_dir}")
    
    return episode_rewards


def evaluate_agent(env, agent, num_episodes=10):
    """
    Evaluate the agent's performance without exploration
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (deterministic)
            action = agent.select_action(np.concatenate([
                state['image_history'].reshape(-1),
                state['state_history'].reshape(-1)
            ]), evaluate=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Check for early termination
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)


def run_trained_agent(model_dir='saved_models', render=True):
    """
    Run a trained agent in the environment
    """
    # Create environment
    env = AirSimCarEnv()
    
    # Determine state and action dimensions
    state_dim = env.seq_length * (3 * np.prod(env.image_shape) + 3)  # 3 cameras + 3 state variables
    action_dim = 2  # steering and throttle
    
    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SAC(state_dim, action_dim, hidden_dim=256, device=device)
    
    # Load trained model
    agent.load(model_dir)
    print(f"Loaded trained model from {model_dir}")
    
    # Run episodes
    num_episodes = 5
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (deterministic)
            action = agent.select_action(np.concatenate([
                state['image_history'].reshape(-1),
                state['state_history'].reshape(-1)
            ]), evaluate=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if requested
            if render:
                center_img = state['images'][0]  # Center camera
                
                # Add telemetry info to the image
                info_text = f"Steering: {action[0]:.4f}, Throttle: {action[1]:.2f}, Speed: {state['car_state'][0]:.2f}"
                cv2.putText(center_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Autonomous Driving', center_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Check for early termination
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    # Clean up
    env.close()
    if render:
        cv2.destroyAllWindows()


def train_mobilevit_supervised(data_dir='collected_data', model_dir='saved_models', batch_size=32, num_epochs=50):
    """
    Train the MobileViT model using supervised learning on collected data
    """
    # Define data transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DrivingDataset(
        csv_file=os.path.join(data_dir, 'driving_log.csv'),
        root_dir=data_dir,
        transform=data_transform
    )
    
    if len(train_dataset) == 0:
        raise ValueError("No valid data samples found. Cannot proceed with training.")
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTemporalModel().to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create directory for saving models
    os.makedirs(model_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"Starting supervised training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch[0].to(device)  # Shape: (batch, C, H, W)
            labels = batch[1][:, 0].to(device)  # Use steering only, shape: (batch,)
            
            # Create dummy states matching (batch, seq_len, state_features)
            seq_len = images.shape[1] if images.ndim == 5 else 1
            states = torch.zeros(images.shape[0], seq_len, 3, device=device)
            
            optimizer.zero_grad()
            outputs = model(images, states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[1][:, 0].to(device)  # steering only
                seq_len = images.shape[1] if images.ndim == 5 else 1
                states = torch.zeros(images.shape[0], seq_len, 3, device=device)
                outputs = model(images, states)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss {best_val_loss:.6f}")
    
    print("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AirSim Autonomous Car Control')
    parser.add_argument('--mode', type=str, default='train_sac', choices=['train_sac', 'train_supervised', 'run'],
                        help='Operation mode: train_sac, train_supervised, or run')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Directory for saving/loading models')
    parser.add_argument('--data_dir', type=str, default='collected_data', help='Directory for collected data')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--render', action='store_true', help='Render environment during execution')
    
    args = parser.parse_args()
    
    if args.mode == 'train_sac':
        # Create environment
        env = AirSimCarEnv()
        
        # Determine state and action dimensions
        state_dim = env.seq_length * (3 * np.prod(env.image_shape) + 3)  # 3 cameras + 3 state variables
        action_dim = 2  # steering and throttle
        
        # Create agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = SAC(state_dim, action_dim, hidden_dim=256, device=device)
        
        # Create replay buffer
        replay_buffer = ReplayBuffer(capacity=1000000)
        
        # Train agent
        train_sac(env, agent, replay_buffer, num_episodes=args.episodes, batch_size=args.batch_size, model_dir=args.model_dir)
        
    elif args.mode == 'train_supervised':
        train_mobilevit_supervised(data_dir=args.data_dir, model_dir=args.model_dir, batch_size=args.batch_size)
        
    elif args.mode == 'run':
        run_trained_agent(model_dir=args.model_dir, render=args.render)
