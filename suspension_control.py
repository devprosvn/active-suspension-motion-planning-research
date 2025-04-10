import numpy as np
from typing import Dict

class SuspensionController:
    """Lớp điều khiển hệ thống treo chủ động"""
    
    def __init__(self):
        # Load simple PID controller (placeholder for actual trained controller)
        self.controller = {
            'Kp': np.array([1.0, 1.0, 0.5]),  # Proportional gains
            'Ki': np.array([0.1, 0.1, 0.05]), # Integral gains
            'Kd': np.array([0.5, 0.5, 0.2]),  # Derivative gains
            'prev_error': np.zeros(3),
            'integral': np.zeros(3)
        }
        
    def get_action(self, impact_prediction: np.ndarray, vehicle_state: Dict) -> np.ndarray:
        """Tính toán hành động điều khiển hệ thống treo"""
        # Tính toán lỗi dựa trên dự đoán tác động
        error = impact_prediction
        
        # Tính toán tích phân lỗi
        self.controller['integral'] += error
        
        # Tính toán đạo hàm lỗi
        derivative = error - self.controller['prev_error']
        self.controller['prev_error'] = error
        
        # Tính toán hành động PID
        action = (
            self.controller['Kp'] * error +
            self.controller['Ki'] * self.controller['integral'] +
            self.controller['Kd'] * derivative
        )
        
        # Giới hạn hành động trong phạm vi cho phép
        return np.clip(action, -1.0, 1.0)
    
    def update_controller(self, new_data: Dict):
        """Cập nhật bộ điều khiển với dữ liệu mới"""
        if 'impact' in new_data and 'action' in new_data:
            # Tính toán lỗi mới
            error = new_data['impact'] - new_data['action']
            
            # Cập nhật tham số PID với learning rate nhỏ
            learning_rate = 0.01
            self.controller['Kp'] += learning_rate * error * new_data['action']
            self.controller['Ki'] += learning_rate * error * self.controller['integral']
            self.controller['Kd'] += learning_rate * error * (error - self.controller['prev_error'])
            
            # Reset tích phân nếu lỗi quá lớn
            if np.linalg.norm(error) > 2.0:
                self.controller['integral'] = np.zeros(3)