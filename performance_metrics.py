import numpy as np
import pandas as pd
from typing import Dict, List, Any

class PerformanceMetrics:
    """Class to collect and analyze performance metrics from Town04 simulations"""
    
    def __init__(self):
        self.metrics_data = []
    
    def record_metrics(self, 
                     scenario_name: str, 
                     vehicle_data: Dict[str, Any], 
                     suspension_data: Dict[str, Any]) -> None:
        """Record performance metrics for a simulation step"""
        metrics = {
            'scenario': scenario_name,
            'timestamp': vehicle_data['timestamp'],
            'speed': vehicle_data['speed'],
            'acceleration': vehicle_data['acceleration'],
            'roll': vehicle_data['roll'],
            'pitch': vehicle_data['pitch'],
            'suspension_impact': np.mean(suspension_data['impact']),
            'comfort_score': -np.sum(np.abs(suspension_data['impact'])),
            'stability_score': -(np.abs(vehicle_data['roll']) + np.abs(vehicle_data['pitch']))
        }
        self.metrics_data.append(metrics)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert recorded metrics to pandas DataFrame"""
        return pd.DataFrame(self.metrics_data)
    
    def compare_with_passive(self, passive_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare active suspension performance with passive system"""
        active_data = self.get_metrics_dataframe()
        
        comparison = {
            'comfort_improvement': np.mean(active_data['comfort_score']) - np.mean(passive_data['comfort_score']),
            'stability_improvement': np.mean(active_data['stability_score']) - np.mean(passive_data['stability_score']),
            'impact_reduction': np.mean(active_data['suspension_impact']) - np.mean(passive_data['suspension_impact'])
        }
        
        return comparison
    
    def save_to_csv(self, file_path: str) -> None:
        """Save metrics data to CSV file"""
        df = self.get_metrics_dataframe()
        df.to_csv(file_path, index=False)
    
    def load_from_csv(self, file_path: str) -> None:
        """Load metrics data from CSV file"""
        df = pd.read_csv(file_path)
        self.metrics_data = df.to_dict('records')