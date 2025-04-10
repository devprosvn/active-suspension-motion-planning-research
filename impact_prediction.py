import numpy as np
from typing import Dict

class ImpactPredictor:
    """Lớp dự đoán tác động của địa hình lên hệ thống treo"""
    
    def __init__(self):
        # Load simple linear regression model (placeholder for actual trained model)
        self.model = {
            'weights': np.array([0.3, 0.5, 0.2]),  # Weights for roughness, slope, obstacles
            'bias': 0.1
        }
        
    def predict_impact(self, terrain_data: Dict, vehicle_data: Dict) -> np.ndarray:
        """Dự đoán tác động của địa hình lên hệ thống treo"""
        # Tính toán tác động dựa trên địa hình và trạng thái xe
        impact = np.array([
            terrain_data['roughness'] * self.model['weights'][0],
            terrain_data['slope'] * self.model['weights'][1],
            len(terrain_data['obstacles']) * self.model['weights'][2]
        ]) + self.model['bias']
        
        # Áp dụng hệ số tốc độ
        speed_factor = min(1.0, vehicle_data.get('speed', 0) / 20.0)
        return impact * speed_factor
    
    def update_model(self, new_data: Dict):
        """Cập nhật mô hình dự đoán với dữ liệu mới"""
        # Simple online learning with moving average
        if 'impact' in new_data and 'terrain' in new_data:
            new_weights = np.array([
                new_data['terrain']['roughness'] * new_data['impact'][0],
                new_data['terrain']['slope'] * new_data['impact'][1],
                len(new_data['terrain']['obstacles']) * new_data['impact'][2]
            ])
            
            # Update model weights with learning rate 0.1
            self.model['weights'] = 0.9 * self.model['weights'] + 0.1 * new_weights
            self.model['bias'] = 0.9 * self.model['bias'] + 0.1 * np.mean(new_data['impact'])