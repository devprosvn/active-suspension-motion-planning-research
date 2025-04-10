import carla
import numpy as np
import pandas as pd
from datetime import datetime

class VehicleDataCollector:
    def __init__(self, world, vehicle):
        """
        Initialize data collector for a vehicle in CARLA
        Args:
            world: CARLA world object
            vehicle: CARLA vehicle actor
        """
        self.world = world
        self.vehicle = vehicle
        self.data = []
        
    def collect_data(self, terrain_type):
        """
        Collect vehicle state data including:
        - Velocity
        - Acceleration
        - Steering angle
        - Suspension response (body acceleration, roll angle)
        - Timestamp
        - Terrain type
        """
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        control = self.vehicle.get_control()
        
        # Calculate body acceleration and roll angle (simplified)
        transform = self.vehicle.get_transform()
        roll = transform.rotation.roll
        
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'terrain_type': terrain_type,
            'velocity_x': velocity.x,
            'velocity_y': velocity.y,
            'velocity_z': velocity.z,
            'acceleration_x': acceleration.x,
            'acceleration_y': acceleration.y,
            'acceleration_z': acceleration.z,
            'steering_angle': control.steer,
            'body_roll': roll,
            'throttle': control.throttle,
            'brake': control.brake
        }
        
        self.data.append(data_point)
        return data_point
    
    def save_to_csv(self, filename):
        """Save collected data to CSV file"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        return df