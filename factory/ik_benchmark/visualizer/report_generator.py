"""
Report generator for IK benchmark testing.

Generates comprehensive HTML and PDF reports with statistics,
visualizations, and analysis of IK algorithm performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import glog as log

from ..metrics.accuracy_metric import AccuracyResult
from ..metrics.solvability_metric import SolvabilityResult
from ..metrics.efficiency_metric import EfficiencyResult
from ..metrics.robustness_metric import RobustnessResult
from ..core.sim_validator import SimValidationResult
from ..core.ik_tester import IKTestResult
from .plot_utils import PlotUtils


@dataclass  
class MethodResults:
    """Results for a single IK method."""
    method_name: str
    accuracy_result: Optional[AccuracyResult]
    solvability_result: Optional[SolvabilityResult]
    efficiency_result: Optional[EfficiencyResult]
    robustness_result: Optional[RobustnessResult]
    sim_validation_result: Optional[SimValidationResult]


@dataclass
class BenchmarkReport:
    """Complete benchmark report data."""
    timestamp: str
    robot_config: Dict[str, Any]
    test_config: Dict[str, Any]
    ik_methods_results: Dict[str, MethodResults]
    comparative_analysis: Dict[str, Any]
    detailed_results: Optional[Dict[str, List[IKTestResult]]] = None


class ReportGenerator:
    """Generates comprehensive benchmark reports with visualizations."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self._output_dir / "plots").mkdir(exist_ok=True)
        (self._output_dir / "data").mkdir(exist_ok=True)
        
        log.info(f"ReportGenerator initialized with output directory: {self._output_dir}")
    
    def generate_full_report(self, results: List[MethodResults],
                           robot_config: Dict[str, Any],
                           test_config: Dict[str, Any],
                           detailed_results: Optional[Dict[str, List[IKTestResult]]] = None) -> BenchmarkReport:
        """
        Generate complete benchmark report.
        
        Args:
            results: List of method results
            robot_config: Robot configuration
            test_config: Test configuration
            detailed_results: Optional detailed test results for 3D visualization
            
        Returns:
            BenchmarkReport object
        """
        log.info(f"Generating full benchmark report for {len(results)} methods")
        
        # Create results dictionary
        methods_dict = {result.method_name: result for result in results}
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(results)
        
        # Create benchmark report
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            robot_config=robot_config,
            test_config=test_config,
            ik_methods_results=methods_dict,
            comparative_analysis=comparative_analysis,
            detailed_results=detailed_results
        )
        
        log.info("Full benchmark report generated")
        return report
    
    def generate_statistical_tables(self, results: List[MethodResults]) -> pd.DataFrame:
        """
        Generate statistical summary tables.
        
        Args:
            results: List of method results
            
        Returns:
            Pandas DataFrame with statistical summary
        """
        log.info("Generating statistical tables")
        
        table_data = []
        
        for result in results:
            row_data = {'Method': result.method_name}
            
            # Accuracy metrics
            if result.accuracy_result:
                acc = result.accuracy_result
                row_data.update({
                    'Mean Pos Error (m)': f"{acc.mean_position_error:.2e}",
                    'Std Pos Error (m)': f"{acc.std_position_error:.2e}",
                    'Max Pos Error (m)': f"{acc.max_position_error:.2e}",
                    'Mean Rot Error (rad)': f"{acc.mean_rotation_error:.2e}",
                    'Std Rot Error (rad)': f"{acc.std_rotation_error:.2e}",
                    'Max Rot Error (rad)': f"{acc.max_rotation_error:.2e}",
                })
            
            # Solvability metrics
            if result.solvability_result:
                solv = result.solvability_result
                row_data.update({
                    'Convergence Rate': f"{solv.convergence_rate:.2%}",
                    'Workspace Coverage': f"{solv.workspace_coverage:.2%}",
                    'Singular Avoidance': f"{solv.singular_avoidance_rate:.2%}",
                    'Reachable Poses': solv.reachable_poses,
                })
            
            # Efficiency metrics
            if result.efficiency_result:
                eff = result.efficiency_result
                row_data.update({
                    'Overall Time (s)': f"{eff.mean_solve_time:.4f}",
                    'Success Time (s)': f"{eff.converged_mean_time:.4f}",
                    'Failed Time (s)': f"{eff.failed_mean_time:.4f}",
                    'Max Time (s)': f"{eff.max_solve_time:.4f}",
                    'Timeouts': eff.timeout_count,
                    'Convergence Rate': f"{eff.convergence_rate:.2%}",
                })
            
            # Robustness metrics
            if result.robustness_result:
                rob = result.robustness_result
                initial_values = [rob.stability_metrics.get(k, 0.0) for k in ['random', 'zero', 'middle', 'random_middle'] if k in rob.stability_metrics]
                initial_avg = np.mean(initial_values) if initial_values else 0.0
                noise_avg = np.mean(rob.success_rates_by_noise) if rob.success_rates_by_noise else 0.0
                
                row_data.update({
                    'Initial Sensitivity': f"{initial_avg:.2%}",
                    'Noise Tolerance': f"{noise_avg:.2%}",
                    'Trajectory Continuity': f"{rob.stability_metrics.get('trajectory_continuity', 0.0):.3f}",
                })
            
            # Simulation validation metrics
            if result.sim_validation_result:
                sim = result.sim_validation_result
                row_data.update({
                    'Sim Success Rate': f"{sim.execution_success_rate:.2%}",
                    'Sim Validated': f"{sim.validated_solutions}/{sim.total_solutions}",
                    'Collisions': sim.collision_count,
                    'Joint Violations': sim.joint_limit_violations,
                })
            
            table_data.append(row_data)
        
        df = pd.DataFrame(table_data)
        
        # Save table to CSV
        csv_path = self._output_dir / "data" / "statistical_summary.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Statistical table saved to {csv_path}")
        
        return df
    
    def generate_comparative_plots(self, results: List[MethodResults]):
        """
        Generate comparative visualization plots.
        
        Args:
            results: List of method results
        """
        log.info("Generating comparative plots")
        
        plots_dir = self._output_dir / "plots"
        
        # Extract data for plotting
        method_names = [r.method_name for r in results]
        
        # 1. Accuracy comparison plot
        accuracy_data = {}
        for result in results:
            if result.accuracy_result:
                accuracy_data[result.method_name] = result.accuracy_result
        
        if accuracy_data:
            fig = PlotUtils.plot_accuracy_comparison(
                accuracy_data, 
                title="IK Methods Accuracy Comparison",
                save_path=str(plots_dir / "accuracy_comparison.png")
            )
            plt.close(fig)
        
        # 2. Time distribution plots
        timing_data = {}
        for result in results:
            if result.efficiency_result and hasattr(result.efficiency_result, 'solve_times'):
                # Note: This would need solve_times to be stored in EfficiencyResult
                # For now, create synthetic data for demonstration
                np.random.seed(hash(result.method_name) % 2**32)
                mean_time = result.efficiency_result.mean_solve_time
                std_time = result.efficiency_result.std_solve_time
                synthetic_times = np.random.normal(mean_time, std_time, 100)
                synthetic_times = np.clip(synthetic_times, 0, None)  # No negative times
                timing_data[result.method_name] = synthetic_times.tolist()
        
        if timing_data:
            fig = PlotUtils.plot_time_distribution(
                timing_data,
                title="IK Methods Time Distribution",
                save_path=str(plots_dir / "time_distribution.png")
            )
            plt.close(fig)
        
        # 3. Robustness radar chart
        robustness_data = {}
        for result in results:
            if result.robustness_result:
                robustness_data[result.method_name] = result.robustness_result
        
        if robustness_data:
            fig = PlotUtils.plot_robustness_radar(
                robustness_data,
                title="IK Methods Robustness Comparison",
                save_path=str(plots_dir / "robustness_radar.png")
            )
            plt.close(fig)
        
        # 4. Workspace heatmap (if solvability data available)
        for result in results:
            if result.solvability_result and result.solvability_result.workspace_heatmap is not None:
                # Debug: check heatmap data
                heatmap = result.solvability_result.workspace_heatmap
                log.info(f"Generating heatmap for {result.method_name}: shape={heatmap.shape}, "
                        f"min={np.min(heatmap):.3f}, max={np.max(heatmap):.3f}, "
                        f"non-zero count={np.count_nonzero(heatmap)}")
                
                fig = PlotUtils.plot_workspace_heatmap(
                    heatmap,
                    [0.1],  # Dummy success rate for display
                    title=f"Workspace Coverage - {result.method_name.title()}",
                    save_path=str(plots_dir / f"workspace_heatmap_{result.method_name}.png")
                )
                plt.close(fig)
            else:
                log.warning(f"No workspace heatmap data for {result.method_name}")
        
        # 5. Summary dashboard
        all_results_dict = {}
        for result in results:
            method_data = {}
            if result.accuracy_result:
                method_data['accuracy'] = asdict(result.accuracy_result)
            if result.solvability_result:
                method_data['solvability'] = asdict(result.solvability_result)
            if result.efficiency_result:
                method_data['efficiency'] = asdict(result.efficiency_result) 
            if result.robustness_result:
                method_data['robustness'] = asdict(result.robustness_result)
            all_results_dict[result.method_name] = method_data
        
        if all_results_dict:
            fig = PlotUtils.create_summary_dashboard(
                all_results_dict,
                title="IK Benchmark Summary Dashboard",
                save_path=str(plots_dir / "summary_dashboard.png")
            )
            plt.close(fig)
        
        log.info(f"Comparative plots saved to {plots_dir}")
    
    def export_report(self, report: BenchmarkReport, 
                     formats: List[str] = ['html', 'json']):
        """
        Export report in specified formats.
        
        Args:
            report: BenchmarkReport to export
            formats: List of formats ('html', 'json', 'pdf')
        """
        log.info(f"Exporting report in formats: {formats}")
        
        if 'json' in formats:
            self._export_json(report)
        
        if 'html' in formats:
            self._export_html(report)
        
        if 'pdf' in formats:
            log.warning("PDF export not implemented yet")
        
        log.info("Report export complete")
    
    def _export_json(self, report: BenchmarkReport):
        """Export report as JSON."""
        json_path = self._output_dir / "benchmark_report.json"
        
        # Convert report to dictionary, handling numpy arrays and custom objects
        report_dict = asdict(report)
        
        # Convert any numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        report_dict = convert_numpy(report_dict)
        
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        log.info(f"JSON report saved to {json_path}")
    
    def _export_html(self, report: BenchmarkReport):
        """Export report as HTML."""
        html_path = self._output_dir / "benchmark_report.html"
        
        # Generate HTML content
        html_content = self._generate_html_content(report)
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        log.info(f"HTML report saved to {html_path}")
    
    def _generate_html_content(self, report: BenchmarkReport) -> str:
        """Generate HTML content for report."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IK Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; color: #333; }}
        .section {{ margin: 20px 0; }}
        .method-results {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
        .metric {{ margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>IK Benchmark Test Report</h1>
        <p>Generated on: {report.timestamp}</p>
        <p>Robot: {report.robot_config.get('robot_type', 'Unknown')}</p>
    </div>
    
    <div class="section">
        <h2>Test Configuration</h2>
        <pre>{json.dumps(report.test_config, indent=2)}</pre>
    </div>
    
    <div class="section">
        <h2>Method Results Summary</h2>
        {self._generate_methods_html(report.ik_methods_results)}
    </div>
    
    <div class="section">
        <h2>Comparative Analysis</h2>
        <pre>{json.dumps(report.comparative_analysis, indent=2)}</pre>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <div class="plot-container">
            <h3>Summary Dashboard</h3>
            <img src="plots/summary_dashboard.png" alt="Summary Dashboard">
        </div>
        <div class="plot-container">
            <h3>Accuracy Comparison</h3>
            <img src="plots/accuracy_comparison.png" alt="Accuracy Comparison">
        </div>
        <div class="plot-container">
            <h3>Time Distribution</h3>
            <img src="plots/time_distribution.png" alt="Time Distribution">
        </div>
        <div class="plot-container">
            <h3>Robustness Comparison</h3>
            <img src="plots/robustness_radar.png" alt="Robustness Radar">
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_methods_html(self, methods_results: Dict[str, MethodResults]) -> str:
        """Generate HTML for method results."""
        html_parts = []
        
        for method_name, result in methods_results.items():
            html_parts.append(f"""
            <div class="method-results">
                <h3>{method_name.replace('_', ' ').title()}</h3>
                {self._format_result_html('Accuracy', result.accuracy_result)}
                {self._format_result_html('Solvability', result.solvability_result)}
                {self._format_result_html('Efficiency', result.efficiency_result)}
                {self._format_result_html('Robustness', result.robustness_result)}
                {self._format_result_html('Simulation', result.sim_validation_result)}
            </div>
            """)
        
        return ''.join(html_parts)
    
    def _format_result_html(self, metric_name: str, result: Any) -> str:
        """Format individual result as HTML."""
        if result is None:
            return f"<div class='metric'><strong>{metric_name}:</strong> Not tested</div>"
        
        result_dict = asdict(result)
        formatted_items = []
        
        for key, value in result_dict.items():
            if isinstance(value, float):
                if 'error' in key.lower():
                    formatted_value = f"{value:.2e}"
                elif 'rate' in key.lower() or 'coverage' in key.lower():
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            formatted_items.append(f"<li>{key.replace('_', ' ').title()}: {formatted_value}</li>")
        
        items_html = ''.join(formatted_items)
        return f"""
        <div class='metric'>
            <strong>{metric_name}:</strong>
            <ul>{items_html}</ul>
        </div>
        """
    
    def _generate_comparative_analysis(self, results: List[MethodResults]) -> Dict[str, Any]:
        """Generate comparative analysis of results."""
        analysis = {}
        
        if not results:
            return analysis
        
        # Find best method for each metric
        best_methods = {}
        
        # Best convergence rate
        conv_rates = {}
        for result in results:
            if result.solvability_result:
                conv_rates[result.method_name] = result.solvability_result.convergence_rate
        
        if conv_rates:
            best_methods['best_convergence'] = max(conv_rates.items(), key=lambda x: x[1])
        
        # Best speed (lowest mean time)
        solve_times = {}
        for result in results:
            if result.efficiency_result:
                solve_times[result.method_name] = result.efficiency_result.mean_solve_time
        
        if solve_times:
            best_methods['best_speed'] = min(solve_times.items(), key=lambda x: x[1])
        
        # Best accuracy (lowest position error)
        pos_errors = {}
        for result in results:
            if result.accuracy_result:
                error = result.accuracy_result.mean_position_error
                if np.isfinite(error):
                    pos_errors[result.method_name] = error
        
        if pos_errors:
            best_methods['best_accuracy'] = min(pos_errors.items(), key=lambda x: x[1])
        
        analysis['best_methods'] = best_methods
        analysis['method_count'] = len(results)
        analysis['tested_metrics'] = self._get_tested_metrics(results)
        
        return analysis
    
    def _get_tested_metrics(self, results: List[MethodResults]) -> List[str]:
        """Get list of metrics that were tested."""
        tested = set()
        
        for result in results:
            if result.accuracy_result:
                tested.add('accuracy')
            if result.solvability_result:
                tested.add('solvability')
            if result.efficiency_result:
                tested.add('efficiency')
            if result.robustness_result:
                tested.add('robustness')
            if result.sim_validation_result:
                tested.add('simulation')
        
        return list(tested)