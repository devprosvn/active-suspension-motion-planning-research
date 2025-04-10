import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Any
from suspension_predictor import SuspensionPredictor

class SuspensionEnv(gym.Env):
    """Custom environment for vehicle suspension control in CARLA"""
    
    def __init__(self, carla_world, vehicle, terrain_detector):
        super(SuspensionEnv, self).__init__()
        
        self.world = carla_world
        self.vehicle = vehicle
        self.predictor = SuspensionPredictor(carla_world, vehicle, terrain_detector)
        
        # State space: [predicted_impact, current_suspension_state, vehicle_state]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # Adjust based on actual state dimensions
            dtype=np.float32
        )
        
        # Action space: [spring_stiffness, damping_force, ride_height]
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5, -0.1]),  # Min values
            high=np.array([1.5, 1.5, 0.1]),   # Max values (normalized)
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment
        Args:
            action: Action to apply to suspension system
        Returns:
            observation: New state observation
            reward: Calculated reward
            done: Whether episode is done
            info: Additional information
        """
        # Apply action to vehicle suspension
        self._apply_suspension_action(action)
        
        # Get new state
        prediction = self.predictor.collect_and_predict()
        self.state = self._get_state_from_prediction(prediction)
        
        # Calculate reward
        reward = self._calculate_reward(prediction)
        
        # Check termination conditions
        done = self._check_termination()
        
        return self.state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Reset vehicle to default suspension settings
        self._reset_suspension()
        
        # Get initial state
        prediction = self.predictor.collect_and_predict()
        self.state = self._get_state_from_prediction(prediction)
        
        return self.state
    
    def _apply_suspension_action(self, action: np.ndarray):
        """Apply action to vehicle suspension system"""
        # Áp dụng hành động điều khiển lên hệ thống treo xe
        physics_control = self.vehicle.get_physics_control()
        
        # Cập nhật thông số hệ thống treo
        wheels = physics_control.wheels
        for i, wheel in enumerate(wheels):
            wheel.suspension_spring_stiffness = action[0] * 50000  # Độ cứng lò xo
            wheel.suspension_damping = action[1] * 5000            # Lực giảm chấn
            wheel.suspension_max_compression = 0.1 + action[2] * 0.05  # Độ nén tối đa
        
        physics_control.wheels = wheels
        self.vehicle.apply_physics_control(physics_control)
        self.last_action = action
    
    def _get_state_from_prediction(self, prediction: Dict[str, Any]) -> np.ndarray:
        """Convert prediction dictionary to state vector"""
        # Extract relevant features from prediction
        impact = prediction['predicted_impact']
        vehicle_data = prediction['vehicle_data']
        
        # Create state vector
        state = np.concatenate([
            impact,  # Predicted impact values
            [vehicle_data['speed'], vehicle_data['acceleration']],  # Vehicle state
            self.last_action if hasattr(self, 'last_action') else np.zeros(3)  # Last action
        ])
        
        return state.astype(np.float32)
    
    def _calculate_reward(self, prediction: Dict[str, Any]) -> float:
        """Calculate reward based on prediction and vehicle state"""
        impact = prediction['predicted_impact']
        vehicle_data = prediction['vehicle_data']
        
        # Reward components
        comfort_reward = -np.sum(np.abs(impact))  # Minimize suspension impact
        stability_reward = -np.abs(vehicle_data['roll']) - np.abs(vehicle_data['pitch'])
        speed_penalty = -0.1 * vehicle_data['speed']  # Encourage moderate speed
        
        # Combine rewards with weights
        total_reward = 0.7 * comfort_reward + 0.2 * stability_reward + 0.1 * speed_penalty
        
        return float(total_reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Kiểm tra va chạm
        if len(self.vehicle.get_collision_history()) > 0:
            return True
            
        # Kiểm tra góc nghiêng quá lớn
        transform = self.vehicle.get_transform()
        roll = abs(transform.rotation.roll)
        pitch = abs(transform.rotation.pitch)
        if roll > 45 or pitch > 30:
            return True
            
        # Kiểm tra tốc độ quá thấp (xe bị kẹt)
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if speed < 1.0 and self.vehicle.get_control().throttle > 0.5:
            return True
            
        return False
    
    def _reset_suspension(self):
        """Reset suspension to default settings"""
        physics_control = self.vehicle.get_physics_control()
        
        # Reset về giá trị mặc định
        wheels = physics_control.wheels
        for wheel in wheels:
            wheel.suspension_spring_stiffness = 50000
            wheel.suspension_damping = 5000
            wheel.suspension_max_compression = 0.1
        
        physics_control.wheels = wheels
        self.vehicle.apply_physics_control(physics_control)