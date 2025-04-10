import carla
import random
from typing import List, Dict, Any

class EvaluationScenarios:
    """Class to setup diverse driving scenarios on Town04 for suspension system evaluation"""
    
    def __init__(self, carla_world):
        self.world = carla_world
        self.map = self.world.get_map()
        
    def setup_town04_scenarios(self) -> List[Dict[str, Any]]:
        """Setup various test scenarios on Town04 map"""
        scenarios = []
        
        # Get all spawn points on Town04
        spawn_points = self.map.get_spawn_points()
        
        # Scenario 1: Flat terrain with varying speeds
        scenarios.append({
            'name': 'flat_terrain',
            'description': 'Flat terrain with speed variations',
            'waypoints': self._generate_route(spawn_points[10], spawn_points[50]),
            'terrain_params': {'roughness': 0.0},
            'speed_profile': [30, 50, 70],  # km/h
            'environment': {'wind': False}
        })
        
        # Scenario 2: Rough terrain with obstacles
        scenarios.append({
            'name': 'rough_terrain',
            'description': 'Rough terrain with obstacles',
            'waypoints': self._generate_route(spawn_points[30], spawn_points[80]),
            'terrain_params': {'roughness': 0.8},
            'speed_profile': [20, 30],
            'environment': {'wind': True, 'wind_strength': 0.5}
        })
        
        # Scenario 3: Mixed terrain with elevation changes
        scenarios.append({
            'name': 'mixed_terrain',
            'description': 'Mixed terrain with elevation changes',
            'waypoints': self._generate_route(spawn_points[5], spawn_points[65]),
            'terrain_params': {'roughness': 0.5},
            'speed_profile': [40, 60],
            'environment': {'wind': False}
        })
        
        return scenarios
    
    def _generate_route(self, start: carla.Transform, end: carla.Transform) -> List[carla.Waypoint]:
        """Generate route between two points on Town04"""
        start_waypoint = self.map.get_waypoint(start.location)
        end_waypoint = self.map.get_waypoint(end.location)
        return self.map.generate_waypoints(start_waypoint, end_waypoint, resolution=1.0)
    
    def apply_terrain_parameters(self, scenario: Dict[str, Any]):
        """Apply terrain parameters to the world"""
        # Apply terrain roughness
        roughness = scenario['terrain_params']['roughness']
        
        # Get all static mesh actors in the world
        static_meshes = self.world.get_actors().filter('static.*')
        
        # Apply physics parameters based on roughness
        for actor in static_meshes:
            physics_control = actor.get_physics_control()
            physics_control.friction = 0.6 + (roughness * 0.4)
            physics_control.damping = 0.1 + (roughness * 0.9)
            actor.apply_physics_control(physics_control)
    
    def apply_environment_effects(self, scenario: Dict[str, Any]):
        """Apply environmental effects like wind"""
        env_params = scenario['environment']
        
        # Apply wind effects if enabled
        if env_params.get('wind', False):
            wind_strength = env_params.get('wind_strength', 0.5)
            weather = self.world.get_weather()
            weather.wind_intensity = wind_strength * 100  # Scale to CARLA units
            self.world.set_weather(weather)
            
            # Apply wind force to all vehicles
            vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                # Calculate wind direction (using map orientation)
                wind_direction = carla.Vector3D(
                    x=random.uniform(-1, 1),
                    y=random.uniform(-1, 1),
                    z=0
                ).make_unit_vector()
                
                # Apply wind force
                vehicle.add_force(wind_direction * wind_strength * 1000)