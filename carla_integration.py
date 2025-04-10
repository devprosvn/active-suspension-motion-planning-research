import carla
import time
import numpy as np
from evaluation_scenarios import EvaluationScenarios
from performance_metrics import PerformanceMetrics
from suspension_env import SuspensionEnv

class CarlaIntegration:
    """Class to integrate CARLA simulation with active suspension system on Town04"""
    
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Setup Town04 map
        self._load_town04()
        
        # Initialize evaluation scenarios
        self.scenario_manager = EvaluationScenarios(self.world)
        
        # Initialize performance metrics
        self.metrics = PerformanceMetrics()
        
    def _load_town04(self):
        """Load Town04 map if not already loaded"""
        if self.world.get_map().name != 'Town04':
            self.client.load_world('Town04')
            self.world = self.client.get_world()
    
    def run_evaluation(self):
        """Run all evaluation scenarios on Town04"""
        # Get all scenarios
        scenarios = self.scenario_manager.setup_town04_scenarios()
        
        # Spawn vehicle
        vehicle = self._spawn_vehicle()
        
        # Initialize all components for evaluation
        from terrain_detection import TerrainDetector
        from impact_prediction import ImpactPredictor
        from suspension_control import SuspensionController
        
        # Stage 1: Terrain detection
        terrain_detector = TerrainDetector(self.world, vehicle)
        
        # Stage 2: Impact prediction
        impact_predictor = ImpactPredictor()
        
        # Stage 3: Suspension control
        suspension_controller = SuspensionController()
        
        suspension_env = SuspensionEnv(self.world, vehicle, terrain_detector)
        
        for scenario in scenarios:
            print(f"Running scenario: {scenario['name']}")
            self._run_scenario(scenario, vehicle, suspension_env)
        
        # Save metrics
        self.metrics.save_to_csv('town04_performance_metrics.csv')
    
    def _spawn_vehicle(self) -> carla.Vehicle:
        """Spawn a vehicle in Town04"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        return self.world.spawn_actor(vehicle_bp, spawn_point)
    
    def _run_scenario(self, scenario: Dict[str, Any], vehicle: carla.Vehicle, suspension_env: SuspensionEnv):
        """Run a single evaluation scenario"""
        # Apply scenario parameters
        self.scenario_manager.apply_terrain_parameters(scenario)
        self.scenario_manager.apply_environment_effects(scenario)
        
        # Run scenario
        for speed in scenario['speed_profile']:
            self._set_vehicle_speed(vehicle, speed)
            
            # Run suspension control loop
            observation = suspension_env.reset()
            done = False
            
            while not done:
                # Get action from suspension controller
                terrain_data = terrain_detector.detect_terrain()
                vehicle_data = suspension_env.predictor.data_collector.collect_data()
                impact_prediction = impact_predictor.predict_impact(terrain_data, vehicle_data)
                action = suspension_controller.get_action(impact_prediction, vehicle_data)
                
                # Step environment
                observation, reward, done, info = suspension_env.step(action)
                
                # Record metrics
                vehicle_data = suspension_env.predictor.data_collector.collect_data()
                suspension_data = {'impact': observation[:3]}
                self.metrics.record_metrics(scenario['name'], vehicle_data, suspension_data)
                
                time.sleep(0.1)
    
    def _set_vehicle_speed(self, vehicle: carla.Vehicle, target_speed: float):
        """Set vehicle to maintain target speed using PID control"""
        # Convert km/h to m/s for CARLA
        target_speed = target_speed / 3.6
        
        # Get current speed
        current_speed = vehicle.get_velocity().length()
        
        # Simple PID controller for speed control
        Kp = 0.5
        Ki = 0.01
        Kd = 0.1
        
        # Calculate error
        error = target_speed - current_speed
        
        # Initialize PID terms if not exists
        if not hasattr(self, '_prev_error'):
            self._prev_error = error
            self._integral = 0
        
        # Calculate PID terms
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error
        
        # Calculate throttle/brake
        control = Kp * error + Ki * self._integral + Kd * derivative
        
        # Apply control to vehicle
        if control > 0:
            vehicle.apply_control(carla.VehicleControl(
                throttle=min(control, 1.0),
                brake=0.0
            ))
        else:
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0,
                brake=min(abs(control), 1.0)
            ))