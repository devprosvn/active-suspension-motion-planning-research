import airsim
import numpy as np
import os
import time
import cv2
import pandas as pd
from datetime import datetime

class DataCollector:
    def __init__(self, save_dir='collected_data'):
        # Connect to AirSim
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
        # Reset car
        self.client.reset()
        
        # Create directory for saving data
        self.save_dir = save_dir
        self.images_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize data log
        self.data_log = []
        
        # Camera names
        self.camera_names = ['center_camera', 'left_camera', 'right_camera']
        
        print("DataCollector initialized. Ready to collect data.")
    
    def collect_frame(self):
        # Get car state
        car_state = self.client.getCarState()
        
        # Get steering, throttle, and speed
        steering = car_state.kinematics_estimated.orientation.z_val
        # In AirSim 1.2.0, we need to get current control inputs through a different approach
        # since getCarControls() and controls attribute may not be available
        # For data collection, we'll note that throttle input isn't directly accessible
        throttle = 0.0  # Placeholder since direct throttle input isn't accessible in this API version
        speed = car_state.speed
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Collect images from all cameras
        images = {}
        camera_id_map = {'center_camera': 0, 'left_camera': 1, 'right_camera': 2}
        for camera_name in self.camera_names:
            camera_id = camera_id_map.get(camera_name, 0)
            response = self.client.simGetImages([airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False)])[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            
            # Calculate dimensions based on actual data size (assuming 3 channels for RGB)
            if img1d.size % 3 == 0:
                total_pixels = img1d.size // 3
                # Try to find reasonable dimensions if possible
                height = response.height if response.height > 0 else int(np.sqrt(total_pixels))
                width = response.width if response.width > 0 else (total_pixels // height)
                if height * width != total_pixels:
                    # Adjust if the dimensions are still incorrect
                    height = int(np.sqrt(total_pixels))
                    width = total_pixels // height
                    if height * width != total_pixels:
                        # Final attempt to make dimensions match
                        for h in range(height, height-100, -1):
                            if total_pixels % h == 0:
                                height = h
                                width = total_pixels // height
                                break
                        if height * width != total_pixels:
                            print(f"Warning: Cannot find valid dimensions for {camera_name}. Skipping frame.")
                            continue
                img_rgb = img1d.reshape(height, width, 3)
            else:
                print(f"Warning: Invalid image data size for {camera_name}. Size {img1d.size} not divisible by 3. Skipping frame.")
                continue
            
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Save image
            img_filename = f"{camera_name}_{timestamp}.png"
            img_path = os.path.join(self.images_dir, img_filename)
            cv2.imwrite(img_path, img_bgr)
            
            images[camera_name] = img_filename
        
        # Check if we have images from all cameras before logging
        if len(images) != len(self.camera_names):
            print("Warning: Could not collect images from all cameras. Skipping this frame.")
            return None
        
        # Log data
        log_entry = {
            'timestamp': timestamp,
            'center_image': images['center_camera'],
            'left_image': images['left_camera'],
            'right_image': images['right_camera'],
            'steering': steering,
            'throttle': throttle,
            'speed': speed
        }
        
        self.data_log.append(log_entry)
        return log_entry
    
    def save_data_log(self):
        # Save data log to CSV
        df = pd.DataFrame(self.data_log)
        csv_path = os.path.join(self.save_dir, 'driving_log.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data log saved to {csv_path}")
    
    def manual_data_collection(self, duration_seconds=300, sample_freq_hz=10):
        """
        Collect data while user manually controls the car using keyboard inputs in AirSim
        """
        print(f"Starting manual data collection for {duration_seconds} seconds...")
        print("Please control the car using keyboard in the AirSim window.")
        print("Press 'q' in the OpenCV window to stop data collection early.")
        
        # Create a window to show the center camera feed
        cv2.namedWindow('Center Camera', cv2.WINDOW_NORMAL)
        
        start_time = time.time()
        sample_interval = 1.0 / sample_freq_hz
        next_sample_time = start_time
        
        try:
            while time.time() - start_time < duration_seconds:
                current_time = time.time()
                
                # Check if it's time to collect a sample
                if current_time >= next_sample_time:
                    # Collect frame data
                    frame_data = self.collect_frame()
                    # Skip display if no frame data was collected
                    if frame_data is None:
                        next_sample_time = current_time + sample_interval
                        continue
                    
                    # Display center camera image
                    center_img_path = os.path.join(self.images_dir, frame_data['center_image'])
                    center_img = cv2.imread(center_img_path)
                    
                    # Add telemetry info to the image
                    info_text = f"Steering: {frame_data['steering']:.4f}, Throttle: {frame_data['throttle']:.2f}, Speed: {frame_data['speed']:.2f}"
                    cv2.putText(center_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Center Camera', center_img)
                    
                    # Calculate next sample time
                    next_sample_time = current_time + sample_interval
                
                # Check for key press to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Data collection stopped early by user.")
                    break
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
        
        finally:
            # Save collected data
            self.save_data_log()
            cv2.destroyAllWindows()
            print("Data collection completed.")
    
    def cleanup(self):
        # Return control to user
        self.client.enableApiControl(False)
        print("Control returned to user. DataCollector cleaned up.")


if __name__ == "__main__":
    # Create data collector
    collector = DataCollector(save_dir='collected_data')
    
    try:
        # Collect data for 5 minutes (300 seconds) at 10Hz
        collector.manual_data_collection(duration_seconds=300, sample_freq_hz=10)
    
    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    
    finally:
        # Clean up
        collector.cleanup()
