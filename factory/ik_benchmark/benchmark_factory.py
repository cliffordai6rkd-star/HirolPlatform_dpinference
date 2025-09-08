"""
IK Benchmark Factory - Main entry point for IK benchmarking system.

Coordinates all benchmark components including data generation, testing,
evaluation, and reporting for comprehensive IK algorithm analysis.
"""

import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import glog as log

from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory

from .core.ik_tester import IKTester
from .core.data_generator import DataGenerator
from .core.sim_validator import SimValidator
from .visualizer.report_generator import ReportGenerator, BenchmarkReport, MethodResults


class IKBenchmarkSuite:
    """Complete benchmark suite for IK methods evaluation."""
    
    def __init__(self, robot_factory: RobotFactory, config: Dict[str, Any]):
        """
        Initialize benchmark suite.
        
        Args:
            robot_factory: Initialized robot factory
            config: Benchmark configuration
        """
        self._robot_factory = robot_factory
        self._config = config
        
        # Get robot model - try different approaches
        robot_model = None
        if hasattr(robot_factory, '_robot_model'):
            robot_model = robot_factory._robot_model
        else:
            # Create robot model using factory config (same as MotionFactory approach)
            from motion.pin_model import RobotModel
            factory_config = robot_factory._config
            model_config = factory_config["model_config"]
            model_name = model_config["name"]
            model_cfg = model_config["cfg"]
            robot_model = RobotModel(model_cfg[model_name])
        
        # Initialize components
        self._ik_tester = IKTester(robot_factory, config)
        self._data_generator = DataGenerator(robot_model, config)
        
        # Initialize simulation validator if enabled
        self._sim_validator = None
        if config.get('simulation', {}).get('validation_enabled', False):
            try:
                self._sim_validator = SimValidator(robot_factory, config.get('simulation', {}))
            except Exception as e:
                log.warning(f"Failed to initialize simulation validator: {e}")
        
        # Initialize report generator
        output_dir = config.get('output', {}).get('directory', 'benchmark_results')
        self._report_generator = ReportGenerator(output_dir)
        
        log.info("IKBenchmarkSuite initialized successfully")
    
    def run_accuracy_test(self, ik_methods: List[str], test_poses: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Run accuracy tests for specified IK methods.
        
        Args:
            ik_methods: List of IK method names
            test_poses: List of target poses
            **kwargs: Additional parameters including:
                - seed: Custom initial joint configuration (np.ndarray)
                - tolerance: Override default tolerance
                - max_iterations: Override default max iterations
        """
        log.info(f"Running accuracy tests for {len(ik_methods)} methods on {len(test_poses)} poses")
        
        results = {}
        tolerance = kwargs.get('tolerance', self._config.get('evaluation', {}).get('tolerance', 1e-6))
        max_iterations = kwargs.get('max_iterations', self._config.get('evaluation', {}).get('max_iterations', 1000))
        
        # Log custom initial_guess usage
        if 'initial_guess' in kwargs:
            log.info(f"Using custom initial guess: {kwargs['initial_guess'][:3]}...") # Show first 3 joints
        
        for method in ik_methods:
            if self._ik_tester.validate_ik_method(method):
                result = self._ik_tester.test_accuracy(method, test_poses, tolerance, 
                                                     max_iterations=max_iterations, **kwargs)
                results[method] = result
            else:
                log.warning(f"IK method {method} not available, skipping")
        
        return results
    
    def run_solvability_test(self, ik_methods: List[str], test_poses: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Run solvability tests for specified IK methods.
        
        Args:
            ik_methods: List of IK method names
            test_poses: List of target poses
            **kwargs: Additional parameters including seed, tolerance, max_iterations
        """
        log.info(f"Running solvability tests for {len(ik_methods)} methods on {len(test_poses)} poses")
        
        results = {}
        tolerance = kwargs.get('tolerance', self._config.get('evaluation', {}).get('tolerance', 1e-6))
        max_iterations = kwargs.get('max_iterations', self._config.get('evaluation', {}).get('max_iterations', 1000))
        
        for method in ik_methods:
            if self._ik_tester.validate_ik_method(method):
                result = self._ik_tester.test_solvability(method, test_poses, tolerance, 
                                                        max_iterations=max_iterations, **kwargs)
                results[method] = result
            else:
                log.warning(f"IK method {method} not available, skipping")
        
        return results
    
    def run_efficiency_test(self, ik_methods: List[str], test_poses: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Run efficiency tests for specified IK methods.
        
        Args:
            ik_methods: List of IK method names
            test_poses: List of target poses
            **kwargs: Additional parameters including seed, tolerance, max_iterations
        """
        log.info(f"Running efficiency tests for {len(ik_methods)} methods on {len(test_poses)} poses")
        
        results = {}
        tolerance = kwargs.get('tolerance', self._config.get('evaluation', {}).get('tolerance', 1e-6))
        max_iterations = kwargs.get('max_iterations', self._config.get('evaluation', {}).get('max_iterations', 1000))
        
        for method in ik_methods:
            if self._ik_tester.validate_ik_method(method):
                result = self._ik_tester.test_efficiency(method, test_poses, tolerance, 
                                                       max_iterations=max_iterations, **kwargs)
                results[method] = result
            else:
                log.warning(f"IK method {method} not available, skipping")
        
        return results
    
    def run_robustness_test(self, ik_methods: List[str], test_poses: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Run robustness tests for specified IK methods."""
        log.info(f"Running robustness tests for {len(ik_methods)} methods on {len(test_poses)} poses")
        
        # Log initial_guess parameter if provided
        if 'initial_guess' in kwargs:
            log.info(f"Using custom initial guess: {kwargs['initial_guess'][:3]}...")
        
        results = {}
        tolerance = self._config.get('evaluation', {}).get('tolerance', 1e-6)
        max_iterations = self._config.get('evaluation', {}).get('max_iterations', 1000)
        noise_levels = self._config.get('evaluation', {}).get('noise_levels', [0.001, 0.005, 0.01, 0.05])
        
        for method in ik_methods:
            if self._ik_tester.validate_ik_method(method):
                result = self._ik_tester.test_robustness(method, test_poses, noise_levels, tolerance, 
                                                       max_iterations=max_iterations, **kwargs)
                results[method] = result
            else:
                log.warning(f"IK method {method} not available, skipping")
        
        return results
    
    def validate_in_simulation(self, solutions: Dict[str, List[np.ndarray]], 
                             target_poses: List[np.ndarray]) -> Dict[str, Any]:
        """Validate IK solutions in simulation."""
        if not self._sim_validator:
            log.warning("Simulation validator not available, skipping simulation validation")
            return {}
        
        log.info(f"Running simulation validation for {len(solutions)} method(s)")
        
        results = {}
        tolerance = self._config.get('simulation', {}).get('validation_tolerance', 1e-3)
        
        for method_name, method_solutions in solutions.items():
            if len(method_solutions) == len(target_poses):
                result = self._sim_validator.validate_ik_solutions(
                    method_solutions, target_poses, tolerance
                )
                results[method_name] = result
            else:
                log.warning(f"Solution count mismatch for {method_name}, skipping simulation validation")
        
        return results


class IKBenchmarkFactory:
    """Main factory for creating and running IK benchmark tests."""
    
    def __init__(self, robot_factory: RobotFactory, config: Dict[str, Any]):
        """
        Initialize IK benchmark factory.
        
        Args:
            robot_factory: Initialized robot factory instance
            config: Benchmark configuration dictionary
        """
        self._robot_factory = robot_factory
        self._config = config
        
        log.info(f"IKBenchmarkFactory initialized for robot: "
                f"{config.get('robot_config', {}).get('robot', 'unknown')}")
    
    def create_benchmark_suite(self) -> IKBenchmarkSuite:
        """
        Create benchmark suite with current configuration.
        
        Returns:
            IKBenchmarkSuite instance
        """
        return IKBenchmarkSuite(self._robot_factory, self._config)
    
    def run_full_benchmark(self, ik_methods: List[str], 
                          test_scenarios: List[str], **kwargs) -> BenchmarkReport:
        """
        Run complete benchmark suite on specified IK methods and scenarios.
        
        Args:
            ik_methods: List of IK method names to test
            test_scenarios: List of test scenario names
            **kwargs: Additional parameters (e.g., initial_guess for custom initial guess)
            
        Returns:
            BenchmarkReport with comprehensive results
        """
        log.info(f"Starting full benchmark: {len(ik_methods)} methods, "
                f"{len(test_scenarios)} scenarios")
        
        # Log initial_guess parameter if provided
        if 'initial_guess' in kwargs:
            log.info(f"Using custom initial guess for full benchmark: {kwargs['initial_guess'][:3]}...")
        
        # Create benchmark suite
        suite = self.create_benchmark_suite()
        
        # Generate test data for all scenarios
        all_test_data = self._generate_test_data(test_scenarios)
        
        # Check if we should collect detailed results for 3D visualization
        collect_detailed = self._config.get('output', {}).get('collect_detailed_results', False)
        detailed_results = {} if collect_detailed else None
        
        # Run all tests
        method_results = []
        
        for method in ik_methods:
            log.info(f"Testing method: {method}")
            
            method_result = MethodResults(
                method_name=method,
                accuracy_result=None,
                solvability_result=None,
                efficiency_result=None,
                robustness_result=None,
                sim_validation_result=None
            )
            
            # Combine all test poses
            combined_poses = []
            for scenario_poses in all_test_data.values():
                combined_poses.extend(scenario_poses)
            
            # Run accuracy test
            try:
                accuracy_results = suite.run_accuracy_test([method], combined_poses, **kwargs)
                method_result.accuracy_result = accuracy_results.get(method)
                
                # Collect detailed results if enabled
                if collect_detailed:
                    detailed_test_results = suite._ik_tester._run_ik_tests(method, combined_poses, 
                                                                         kwargs.get('tolerance', 1e-6), **kwargs)
                    detailed_results[method] = detailed_test_results
                    log.info(f"Collected {len(detailed_test_results)} detailed results for {method}")
            except Exception as e:
                log.error(f"Accuracy test failed for {method}: {e}")
            
            # Run solvability test
            try:
                solvability_results = suite.run_solvability_test([method], combined_poses, **kwargs)
                method_result.solvability_result = solvability_results.get(method)
            except Exception as e:
                log.error(f"Solvability test failed for {method}: {e}")
            
            # Run efficiency test
            try:
                efficiency_results = suite.run_efficiency_test([method], combined_poses, **kwargs)
                method_result.efficiency_result = efficiency_results.get(method)
            except Exception as e:
                log.error(f"Efficiency test failed for {method}: {e}")
            
            # Run robustness test (on subset for performance)
            try:
                robustness_poses = combined_poses[:min(50, len(combined_poses))]
                robustness_results = suite.run_robustness_test([method], robustness_poses, **kwargs)
                method_result.robustness_result = robustness_results.get(method)
            except Exception as e:
                log.error(f"Robustness test failed for {method}: {e}")
            
            method_results.append(method_result)
        
        # Run simulation validation if enabled
        if self._config.get('simulation', {}).get('validation_enabled', False):
            try:
                # Collect solutions from all methods for validation
                # This would require modifying the test methods to return solutions
                log.info("Simulation validation skipped - solution collection not implemented")
            except Exception as e:
                log.error(f"Simulation validation failed: {e}")
        
        # Generate report
        robot_config = {
            'robot_type': self._config.get('robot_config', {}).get('robot', 'unknown'),
            'robot_factory_config': self._robot_factory._config
        }
        
        report = suite._report_generator.generate_full_report(
            method_results, robot_config, self._config, detailed_results
        )
        
        # Generate statistical tables
        suite._report_generator.generate_statistical_tables(method_results)
        
        # Generate plots
        suite._report_generator.generate_comparative_plots(method_results)
        
        # Export report
        output_formats = self._config.get('output', {}).get('formats', ['html', 'json'])
        suite._report_generator.export_report(report, output_formats)
        
        log.info("Full benchmark completed successfully")
        return report
    
    def run_single_test(self, ik_method: str, scenario: str, **kwargs) -> MethodResults:
        """
        Run single test on one IK method and one scenario.
        
        Args:
            ik_method: IK method name
            scenario: Test scenario name
            **kwargs: Additional parameters (e.g., initial_guess for custom initial guess)
            
        Returns:
            MethodResults for the single test
        """
        log.info(f"Running single test: {ik_method} on {scenario}")
        
        # Log initial_guess parameter if provided
        if 'initial_guess' in kwargs:
            log.info(f"Using custom initial guess: {kwargs['initial_guess'][:3]}...")
        
        suite = self.create_benchmark_suite()
        
        # Generate test data for scenario
        test_data = self._generate_test_data([scenario])
        poses = test_data.get(scenario, [])
        
        if not poses:
            log.warning(f"No test data generated for scenario {scenario}")
            return MethodResults(
                method_name=ik_method,
                accuracy_result=None,
                solvability_result=None,
                efficiency_result=None,
                robustness_result=None,
                sim_validation_result=None
            )
        
        # Run tests
        method_result = MethodResults(
            method_name=ik_method,
            accuracy_result=None,
            solvability_result=None,
            efficiency_result=None,
            robustness_result=None,
            sim_validation_result=None
        )
        
        try:
            accuracy_results = suite.run_accuracy_test([ik_method], poses, **kwargs)
            method_result.accuracy_result = accuracy_results.get(ik_method)
        except Exception as e:
            log.error(f"Accuracy test failed: {e}")
        
        try:
            solvability_results = suite.run_solvability_test([ik_method], poses, **kwargs)
            method_result.solvability_result = solvability_results.get(ik_method)
        except Exception as e:
            log.error(f"Solvability test failed: {e}")
        
        try:
            efficiency_results = suite.run_efficiency_test([ik_method], poses, **kwargs)
            method_result.efficiency_result = efficiency_results.get(ik_method)
        except Exception as e:
            log.error(f"Efficiency test failed: {e}")
        
        log.info("Single test completed")
        return method_result
    
    def _generate_test_data(self, test_scenarios: List[str]) -> Dict[str, List[np.ndarray]]:
        """
        Generate test data for specified scenarios.
        
        Args:
            test_scenarios: List of scenario names
            
        Returns:
            Dictionary mapping scenario names to pose lists
        """
        log.info(f"Generating test data for scenarios: {test_scenarios}")
        
        # Create robot model using factory config (same as other places)
        from motion.pin_model import RobotModel
        factory_config = self._robot_factory._config
        model_config = factory_config["model_config"]
        model_name = model_config["name"]
        model_cfg = model_config["cfg"]
        robot_model = RobotModel(model_cfg[model_name])
        data_generator = DataGenerator(robot_model, self._config)
        test_data = {}
        
        for scenario in test_scenarios:
            try:
                poses = self._generate_scenario_poses(data_generator, scenario)
                test_data[scenario] = poses
                log.info(f"Generated {len(poses)} poses for scenario '{scenario}'")
            except Exception as e:
                log.error(f"Failed to generate poses for scenario '{scenario}': {e}")
                test_data[scenario] = []
        
        total_poses = sum(len(poses) for poses in test_data.values())
        log.info(f"Total test poses generated: {total_poses}")
        
        return test_data
    
    def _generate_scenario_poses(self, data_generator: DataGenerator, 
                               scenario: str) -> List[np.ndarray]:
        """Generate poses for a specific test scenario."""
        scenario_config = self._config.get('test_scenarios', {}).get(scenario, {})
        
        if scenario == 'random_sampling':
            count = scenario_config.get('count', 100)
            seed = scenario_config.get('seed', None)
            return data_generator.generate_random_poses(count, seed)
        
        elif scenario == 'trajectory_tests':
            poses = []
            for traj_config in scenario_config:
                traj_type = traj_config.get('type', 'line')
                waypoints = traj_config.get('waypoints', 50)
                # Remove waypoints from config to avoid duplicate argument
                traj_config_clean = {k: v for k, v in traj_config.items() if k != 'waypoints'}
                traj_poses = data_generator.generate_trajectory_poses(traj_type, waypoints, **traj_config_clean)
                poses.extend(traj_poses)
            return poses
        
        elif scenario == 'singular_tests':
            count = scenario_config.get('count', 50)
            margin = scenario_config.get('margin', 0.01)
            return data_generator.generate_singular_poses(count, margin)
        
        elif scenario == 'workspace_grid':
            resolution = tuple(scenario_config.get('resolution', [10, 10, 5]))
            orientation_samples = scenario_config.get('orientation_samples', 4)
            return data_generator.generate_workspace_grid(resolution, orientation_samples)
        
        else:
            log.warning(f"Unknown test scenario: {scenario}")
            return []
    
    @classmethod
    def from_config_file(cls, robot_factory: RobotFactory, 
                        config_file: str) -> 'IKBenchmarkFactory':
        """
        Create benchmark factory from configuration file.
        
        Args:
            robot_factory: Robot factory instance
            config_file: Path to YAML configuration file
            
        Returns:
            IKBenchmarkFactory instance
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        log.info(f"Loaded benchmark configuration from {config_file}")
        return cls(robot_factory, config)
    
    def get_available_ik_methods(self) -> List[str]:
        """
        Get list of available IK methods.
        
        Returns:
            List of IK method names
        """
        suite = self.create_benchmark_suite()
        return suite._ik_tester.get_available_ik_methods()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()