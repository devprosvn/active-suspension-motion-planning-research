import carla
import numpy as np

class TerrainDetector:
    """Lớp phát hiện địa hình trong CARLA"""
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world = world
        self.vehicle = vehicle
        
    def detect_terrain(self) -> dict:
        """Phát hiện thông tin địa hình hiện tại"""
        # Lấy vị trí và hướng của xe
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        # Phân tích địa hình xung quanh
        terrain_data = {
            'roughness': self._calculate_roughness(vehicle_location),
            'slope': self._calculate_slope(vehicle_transform),
            'obstacles': self._detect_obstacles(vehicle_location)
        }
        
        return terrain_data
    
    def _calculate_roughness(self, location: carla.Location) -> float:
        """Tính toán độ gồ ghề của địa hình"""
        # Lấy thông tin địa hình từ bản đồ
        waypoint = self.world.get_map().get_waypoint(location)
        road_type = waypoint.lane_type
        
        # Xác định độ gồ ghề dựa trên loại đường
        roughness_map = {
            carla.LaneType.Driving: 0.2,
            carla.LaneType.Sidewalk: 0.5,
            carla.LaneType.Grass: 0.8,
            carla.LaneType.Shoulder: 0.7
        }
        return roughness_map.get(road_type, 0.5)
    
    def _calculate_slope(self, transform: carla.Transform) -> float:
        """Tính toán độ dốc của địa hình"""
        # Lấy vector hướng của xe
        forward_vector = transform.get_forward_vector()
        
        # Tính góc giữa vector hướng và mặt phẳng ngang (x-z)
        slope_angle = np.arctan2(forward_vector.z, forward_vector.x)
        return np.degrees(slope_angle)
    
    def _detect_obstacles(self, location: carla.Location) -> list:
        """Phát hiện chướng ngại vật xung quanh"""
        # Lấy tất cả actors trong bán kính 10m
        nearby_actors = self.world.get_actors()
        obstacles = []
        
        for actor in nearby_actors:
            if actor.id != self.vehicle.id and actor.get_location().distance(location) < 10:
                if 'vehicle' in actor.type_id or 'walker' in actor.type_id or 'static' in actor.type_id:
                    obstacles.append({
                        'type': actor.type_id,
                        'distance': actor.get_location().distance(location),
                        'location': actor.get_location()
                    })
        return obstacles