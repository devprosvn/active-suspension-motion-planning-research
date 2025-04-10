#!/usr/bin/env python

"""
Data Collection Script for Terrain Recognition and Depth Estimation

This script connects to CARLA 0.9.14 and collects image data and depth maps
from different terrain types in Town10 map.
"""

import os
import sys
import glob
import time
import argparse
import random
import numpy as np
import cv2
from datetime import datetime

# Try to import CARLA module
try:
    # Check if CARLA module is in Python path
    carla_path = glob.glob('l:\\CARLA_0.9.14\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(carla_path)
    import carla
except IndexError:
    print('CARLA module not found. Make sure CARLA is installed correctly.')
    sys.exit(1)

# Define terrain types to collect data for
TERRAIN_TYPES = {
    'asphalt': [],  # Will be populated with locations
    'gravel': [],
    'sand': [],
    'snow': [],
    'bumpy': []
}

# Town10 specific locations for different terrain types
# These are example coordinates - they should be replaced with actual coordinates
# from Town10 where these terrain types exist
TOWN10_TERRAIN_LOCATIONS = {
    'asphalt': [
        carla.Location(x=0, y=0, z=0),  # Replace with actual coordinates
        carla.Location(x=10, y=10, z=0),
    ],
    'gravel': [
        carla.Location(x=20, y=20, z=0),  # Replace with actual coordinates
        carla.Location(x=30, y=30, z=0),
    ],
    'sand': [
        carla.Location(x=40, y=40, z=0),  # Replace with actual coordinates
        carla.Location(x=50, y=50, z=0),
    ],
    'snow': [
        carla.Location(x=60, y=60, z=0),  # Replace with actual coordinates
        carla.Location(x=70, y=70, z=0),
    ],
    'bumpy': [
        carla.Location(x=80, y=80, z=0),  # Replace with actual coordinates
        carla.Location(x=90, y=90, z=0),
    ]
}

# Define the data directory structure
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
IMAGE_DIR = os.path.join(DATASET_ROOT, 'images')
DEPTH_DIR = os.path.join(DATASET_ROOT, 'depth')
LABEL_FILE = os.path.join(DATASET_ROOT, 'labels.csv')

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

# Camera settings
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600


class CarlaSensor:
    """Base class for CARLA sensors"""
    def __init__(self, world, transform, attachment_type):
        self.world = world
        self.transform = transform
        self.attachment_type = attachment_type
        self.sensor = None
        self.data = None
        self._callback_registered = False

    def _set_callback(self, callback):
        if not self._callback_registered:
            self.sensor.listen(callback)
            self._callback_registered = True

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class RGBCamera(CarlaSensor):
    """RGB Camera sensor for collecting image data"""
    def __init__(self, world, transform, attachment_type, width=800, height=600, fov=90):
        super().__init__(world, transform, attachment_type)
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        self.sensor = world.spawn_actor(camera_bp, transform)

    def set_image_callback(self, callback):
        self._set_callback(lambda image: callback(image))


class DepthCamera(CarlaSensor):
    """Depth Camera sensor for collecting depth data"""
    def __init__(self, world, transform, attachment_type, width=800, height=600, fov=90):
        super().__init__(world, transform, attachment_type)
        camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        self.sensor = world.spawn_actor(camera_bp, transform)

    def set_depth_callback(self, callback):
        self._set_callback(lambda image: callback(image))


class DataCollector:
    """Main class for collecting data from CARLA"""
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.town = None
        self.vehicle = None
        self.rgb_camera = None
        self.depth_camera = None
        self.current_terrain = None
        self.image_count = 0
        self.label_file = None
        
        # Initialize label file
        if not os.path.exists(LABEL_FILE):
            with open(LABEL_FILE, 'w') as f:
                f.write('image_id,terrain_type,depth_map_id\n')

    def connect_to_carla(self):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            print(f"Connected to CARLA server version {self.client.get_server_version()}")
            return True
        except Exception as e:
            print(f"Failed to connect to CARLA server: {e}")
            return False

    def load_town04(self):
        """Load Town04 map"""
        try:
            self.world = self.client.get_world()
            if self.world.get_map().name != 'Town04':
                self.world = self.client.load_world('Town04')
                print("Loaded Town04 map")
            else:
                print("Already in Town04 map")
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            return True
        except Exception as e:
            print(f"Failed to load Town04: {e}")
            return False

    def spawn_vehicle(self):
        """Spawn a vehicle for data collection"""
        try:
            # Get a random vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            
            # Find a valid spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("No spawn points found")
                return False
            
            spawn_point = random.choice(spawn_points)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Spawned vehicle {self.vehicle.type_id}")
            
            # Attach cameras to vehicle
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.rgb_camera = RGBCamera(
                self.world, camera_transform, 
                carla.AttachmentType.Rigid, 
                width=IMAGE_WIDTH, height=IMAGE_HEIGHT
            )
            self.depth_camera = DepthCamera(
                self.world, camera_transform, 
                carla.AttachmentType.Rigid, 
                width=IMAGE_WIDTH, height=IMAGE_HEIGHT
            )
            
            # Attach cameras to vehicle
            self.rgb_camera.sensor.set_parent(self.vehicle)
            self.depth_camera.sensor.set_parent(self.vehicle)
            
            # Set callbacks
            self.rgb_camera.set_image_callback(self.process_rgb_image)
            self.depth_camera.set_depth_callback(self.process_depth_image)
            
            return True
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return False

    def process_rgb_image(self, image):
        """Process and save RGB image"""
        if self.current_terrain is None:
            return
        
        # Convert CARLA raw image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_id = f"{self.current_terrain}_{timestamp}_{self.image_count}"
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
        cv2.imwrite(image_path, array)
        
        # Store image ID for later use with depth map
        self.current_image_id = image_id

    def process_depth_image(self, image):
        """Process and save depth image"""
        if self.current_terrain is None or not hasattr(self, 'current_image_id'):
            return
        
        # Convert CARLA depth map to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        
        # Convert depth map to a more usable format
        # CARLA encodes depth as R + G * 256 + B * 256 * 256
        depth_map = np.zeros((image.height, image.width), dtype=np.float32)
        normalized = array.astype(np.float32)
        
        # Depth is encoded in the RGB channels
        depth_map = (
            normalized[:, :, 2] + 
            normalized[:, :, 1] * 256 + 
            normalized[:, :, 0] * 256 * 256
        ) / (256 * 256 * 256 - 1)
        depth_map = depth_map * 1000  # Convert to meters
        
        # Save depth map
        depth_id = self.current_image_id
        depth_path = os.path.join(DEPTH_DIR, f"{depth_id}_depth.png")
        
        # Normalize for visualization (0-255)
        normalized_depth = (depth_map * 255 / np.max(depth_map)).astype('uint8')
        cv2.imwrite(depth_path, normalized_depth)
        
        # Write to label file
        with open(LABEL_FILE, 'a') as f:
            f.write(f"{self.current_image_id},{self.current_terrain},{depth_id}_depth\n")
        
        self.image_count += 1

    def teleport_to_terrain(self, terrain_type):
        """Teleport vehicle to a location with the specified terrain type"""
        if terrain_type not in TOWN10_TERRAIN_LOCATIONS:
            print(f"Unknown terrain type: {terrain_type}")
            return False
        
        locations = TOWN10_TERRAIN_LOCATIONS[terrain_type]
        if not locations:
            print(f"No locations defined for terrain type: {terrain_type}")
            return False
        
        location = random.choice(locations)
        transform = carla.Transform(location, carla.Rotation(pitch=0, yaw=0, roll=0))
        
        try:
            self.vehicle.set_transform(transform)
            self.current_terrain = terrain_type
            print(f"Teleported to {terrain_type} terrain at {location}")
            return True
        except Exception as e:
            print(f"Failed to teleport vehicle: {e}")
            return False

    def collect_data(self):
        """Main data collection loop"""
        if not self.connect_to_carla():
            return
        
        if not self.load_town10():
            return
        
        if not self.spawn_vehicle():
            return
        
        try:
            # Collect data for each terrain type
            for terrain_type in TERRAIN_TYPES.keys():
                print(f"\nCollecting data for {terrain_type} terrain...")
                self.current_terrain = terrain_type
                
                # Teleport to terrain location
                if not self.teleport_to_terrain(terrain_type):
                    continue
                
                # Collect images for this terrain type
                images_per_terrain = self.args.images_per_terrain
                collected = 0
                
                while collected < images_per_terrain:
                    # Tick the world to get new sensor data
                    self.world.tick()
                    
                    # Move the vehicle slightly to get different views
                    control = carla.VehicleControl()
                    control.throttle = 0.5
                    control.steer = random.uniform(-0.3, 0.3)
                    self.vehicle.apply_control(control)
                    
                    # Wait for a few ticks to let the vehicle move
                    for _ in range(5):
                        self.world.tick()
                    
                    collected += 1
                    print(f"Collected {collected}/{images_per_terrain} images for {terrain_type}")
                    
                    # Sleep to avoid overwhelming the system
                    time.sleep(0.1)
            
            print("\nData collection completed!")
            print(f"Total images collected: {self.image_count}")
            print(f"Images saved to: {IMAGE_DIR}")
            print(f"Depth maps saved to: {DEPTH_DIR}")
            print(f"Labels saved to: {LABEL_FILE}")
            
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        finally:
            # Clean up
            if self.rgb_camera:
                self.rgb_camera.destroy()
            if self.depth_camera:
                self.depth_camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            
            # Restore asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)


def main():
    parser = argparse.ArgumentParser(description='Collect data from CARLA for terrain recognition')
    parser.add_argument('--images-per-terrain', type=int, default=100,
                        help='Number of images to collect for each terrain type')
    args = parser.parse_args()
    
    collector = DataCollector(args)
    collector.collect_data()


if __name__ == '__main__':
    main()