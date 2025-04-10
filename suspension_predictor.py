import numpy as np
import pandas as pd
from tft_model import SuspensionTFTModel
from vehicle_data_collector import VehicleDataCollector

class SuspensionPredictor:
    def __init__(self, carla_world, vehicle, terrain_detector, model_path=None):
        """
        Initialize suspension impact predictor
        Args:
            carla_world: CARLA world object
            vehicle: CARLA vehicle actor
            terrain_detector: Terrain detection model
            model_path: Path to pre-trained TFT model (optional)
        """
        self.world = carla_world
        self.vehicle = vehicle
        self.terrain_detector = terrain_detector
        
        # Initialize data collector
        self.data_collector = VehicleDataCollector(self.world, self.vehicle)
        
        # Initialize TFT model
        self.tft_model = SuspensionTFTModel()
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
    
    def collect_and_predict(self):
        """
        Collect current vehicle state, detect terrain, and predict suspension impact
        Returns:
            Dictionary containing terrain type and predicted suspension impact
        """
        # Detect current terrain type
        terrain_type = self.detect_terrain()
        
        # Collect current vehicle state
        vehicle_data = self.data_collector.collect_data(terrain_type)
        
        # Prepare data for prediction
        prediction = self.predict_impact(vehicle_data)
        
        return {
            'terrain_type': terrain_type,
            'predicted_impact': prediction,
            'vehicle_data': vehicle_data
        }
    
    def detect_terrain(self):
        """
        Detect current terrain type using terrain detector
        Returns:
            Detected terrain type (string)
        """
        # Get current camera image
        camera = self.vehicle.get_camera()
        image = camera.get_image()
        
        # Detect terrain type
        terrain_type = self.terrain_detector.predict(image)
        
        return terrain_type
    
    def predict_impact(self, vehicle_data):
        """
        Predict suspension impact using TFT model
        Args:
            vehicle_data: Current vehicle state data
        Returns:
            Predicted suspension impact values
        """
        # Convert data to DataFrame
        df = pd.DataFrame([vehicle_data])
        
        # Add time index
        df['time_idx'] = 0
        
        # Make prediction
        prediction = self.tft_model.predict(df)
        
        return prediction
    
    def load_model(self, model_path):
        """
        Load pre-trained TFT model
        Args:
            model_path: Path to saved model
        """
        self.tft_model = torch.load(model_path)
    
    def save_model(self, model_path):
        """
        Save trained TFT model
        Args:
            model_path: Path to save model
        """
        torch.save(self.tft_model, model_path)