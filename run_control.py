import os
import numpy as np
import torch
import airsim
import time
import cv2
from collections import deque
import argparse

from models.mobilevit import VisionTemporalModel
from airsim_env import SAC, AirSimCarEnv, ReplayBuffer

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
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    
    # Custom dataset for driving data
    class DrivingDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.data_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.data_frame)
        
        def __getitem__(self, idx):
            # Get image paths
            center_img_path = os.path.join(self.root_dir, 'images', self.data_frame.iloc[idx, 1])
            
            # Load images
            center_img = cv2.imread(center_img_path)
            center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            if self.transform:
                center_img = self.transform(center_img)
            
            # Get steering angle
            steering = self.data_frame.iloc[idx, 4]
            
            # Get throttle and speed
            throttle = self.data_frame.iloc[idx, 5]
            speed = self.data_frame.iloc[idx, 6]
            
            return {
                'image': center_img,
                'steering': steering,
                'throttle': throttle,
                'speed': speed
            }
    
    # Set up transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = DrivingDataset(
        csv_file=os.path.join(data_dir, 'driving_log.csv'),
        root_dir=data_dir,
        transform=transform
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTemporalModel(seq_len=1, vision_features=128, state_features=2).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Starting supervised training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Get data
            images = batch['image'].to(device)
            steering = batch['steering'].float().to(device)
            throttle = batch['throttle'].float().to(device)
            speed = batch['speed'].float().to(device)
            
            # Prepare input
            states = torch.stack([throttle, speed], dim=1).unsqueeze(1)  # [batch_size, 1, 2]
            images = images.unsqueeze(1)  # [batch_size, 1, 3, H, W]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, states)
            
            # Compute loss
            loss = criterion(outputs, steering)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                images = batch['image'].to(device)
                steering = batch['steering'].float().to(device)
                throttle = batch['throttle'].float().to(device)
                speed = batch['speed'].float().to(device)
                
                # Prepare input
                states = torch.stack([throttle, speed], dim=1).unsqueeze(1)  # [batch_size, 1, 2]
                images = images.unsqueeze(1)  # [batch_size, 1, 3, H, W]
                
                # Forward pass
                outputs = model(images, states)
                
                # Compute loss
                loss = criterion(outputs, steering)
                val_loss += loss.item()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss/len(val_loader):.6f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'mobilevit_supervised.pth'))
    print(f"Model saved to {os.path.join(model_dir, 'mobilevit_supervised.pth')}")


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
