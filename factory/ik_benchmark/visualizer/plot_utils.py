"""
Plotting utilities for IK benchmark visualization.

Provides static methods for generating various plots and visualizations
for benchmark results analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import glog as log

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


class PlotUtils:
    """Static utility methods for creating benchmark visualization plots."""
    
    @staticmethod
    def plot_workspace_heatmap(workspace_data: np.ndarray, 
                              errors: List[float],
                              title: str = "Workspace Error Distribution",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot workspace error distribution as heatmap.
        
        Args:
            workspace_data: 2D array representing workspace grid
            errors: List of errors corresponding to workspace points
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Check if workspace_data has valid values
        if workspace_data is not None and workspace_data.size > 0 and not np.all(np.isnan(workspace_data)):
            # Create heatmap
            im = ax.imshow(workspace_data, cmap='viridis', origin='lower', 
                          aspect='auto', interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Success Rate', rotation=270, labelpad=20)
        else:
            # No valid workspace data to plot
            ax.text(0.5, 0.5, 'No workspace heatmap data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=16, color='red')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Workspace', fontsize=12)
        ax.set_ylabel('Y Workspace', fontsize=12)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Workspace heatmap saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_convergence_curves(convergence_data: Dict[str, List[float]],
                              title: str = "Convergence Rate Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence rate curves for different methods.
        
        Args:
            convergence_data: Dict mapping method names to convergence rates
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(convergence_data))
        
        for i, (method_name, rates) in enumerate(convergence_data.items()):
            if rates:
                x_values = range(len(rates))
                ax.plot(x_values, rates, marker='o', linewidth=2, 
                       label=method_name.replace('_', ' ').title(), 
                       color=colors[i], markersize=4)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Test Batch', fontsize=12)
        ax.set_ylabel('Convergence Rate', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)
        
        # Add percentage formatting to y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Convergence curves saved to {save_path}")
        
        return fig
    
    @staticmethod  
    def plot_time_distribution(timing_data: Dict[str, List[float]],
                             title: str = "Solve Time Distribution",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot solve time distribution histogram for different methods.
        
        Args:
            timing_data: Dict mapping method names to solve times
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram plot
        ax1 = axes[0]
        colors = sns.color_palette("husl", len(timing_data))
        
        for i, (method_name, times) in enumerate(timing_data.items()):
            if times:
                finite_times = [t for t in times if np.isfinite(t)]
                if finite_times:
                    ax1.hist(finite_times, bins=30, alpha=0.7, 
                            label=method_name.replace('_', ' ').title(),
                            color=colors[i], density=True)
        
        ax1.set_title('Time Distribution Histogram', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Solve Time (seconds)', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        box_data = []
        box_labels = []
        
        for method_name, times in timing_data.items():
            if times:
                finite_times = [t for t in times if np.isfinite(t)]
                if finite_times:
                    box_data.append(finite_times)
                    box_labels.append(method_name.replace('_', ' ').title())
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_title('Time Distribution Box Plot', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Solve Time (seconds)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Time distribution plots saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_trajectory_continuity(joint_trajectories: Dict[str, np.ndarray],
                                 title: str = "Joint Trajectory Continuity",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot joint trajectory continuity curves.
        
        Args:
            joint_trajectories: Dict mapping method names to joint trajectories
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if not joint_trajectories:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trajectory data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Determine subplot layout
        n_methods = len(joint_trajectories)
        n_cols = min(2, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (method_name, trajectory) in enumerate(joint_trajectories.items()):
            ax = axes[i] if i < len(axes) else axes[-1]
            
            if trajectory.ndim == 2:  # Multiple joints
                n_joints = trajectory.shape[1]
                time_steps = np.arange(trajectory.shape[0])
                
                # Plot each joint trajectory
                colors = sns.color_palette("husl", n_joints)
                for j in range(min(n_joints, 7)):  # Limit to 7 joints for clarity
                    ax.plot(time_steps, trajectory[:, j], 
                           label=f'Joint {j+1}', color=colors[j], linewidth=1.5)
                
                ax.set_title(f'{method_name.replace("_", " ").title()}', 
                           fontsize=11, fontweight='bold')
                ax.set_xlabel('Time Step', fontsize=10)
                ax.set_ylabel('Joint Angle (rad)', fontsize=10)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(joint_trajectories), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Trajectory continuity plots saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_robustness_radar(robustness_data: Dict[str, Any],
                            title: str = "Robustness Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot robustness metrics as radar chart.
        
        Args:
            robustness_data: Dict mapping method names to robustness results
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if not robustness_data:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No robustness data available',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Define robustness metrics for radar chart
        metrics = [
            'Initial Sensitivity (Random)',
            'Initial Sensitivity (Zero)', 
            'Initial Sensitivity (Middle)',
            'Noise Tolerance (0.001)',
            'Noise Tolerance (0.01)',
            'Trajectory Continuity'
        ]
        
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(robustness_data))
        
        for i, (method_name, rob_result) in enumerate(robustness_data.items()):
            # Extract metric values (normalize to 0-1 scale)
            values = []
            try:
                # Initial sensitivity metrics
                values.append(rob_result.stability_metrics.get('random', 0.0))
                values.append(rob_result.stability_metrics.get('zero', 0.0))
                values.append(rob_result.stability_metrics.get('middle', 0.0))
                
                # Noise tolerance metrics - use success rates by noise level
                noise_levels = rob_result.noise_levels if rob_result.noise_levels else []
                success_rates = rob_result.success_rates_by_noise if rob_result.success_rates_by_noise else []
                
                # Find values for specific noise levels
                for target_level in [0.001, 0.01]:
                    found_value = 0.0
                    if noise_levels and success_rates:
                        for level, rate in zip(noise_levels, success_rates):
                            if abs(level - target_level) < 1e-6:
                                found_value = rate
                                break
                    values.append(found_value)
                
                # Trajectory continuity
                values.append(rob_result.stability_metrics.get('trajectory_continuity', 0.0))
                
            except AttributeError:
                # Handle case where robustness result doesn't have expected structure
                values = [0.0] * n_metrics
                log.warning(f"Invalid robustness data structure for {method_name}")
            
            values += values[:1]  # Complete the circle
            
            # Plot the radar chart
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=method_name.replace('_', ' ').title(), 
                   color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Customize the radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title(title, size=14, fontweight='bold', pad=20)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Robustness radar chart saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_accuracy_comparison(accuracy_data: Dict[str, Any],
                               title: str = "Accuracy Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot accuracy metrics comparison.
        
        Args:
            accuracy_data: Dict mapping method names to accuracy results
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = list(accuracy_data.keys())
        colors = sns.color_palette("husl", len(methods))
        
        # Position errors
        pos_means = []
        pos_stds = []
        rot_means = []
        rot_stds = []
        
        for method_name, acc_result in accuracy_data.items():
            try:
                pos_means.append(acc_result.mean_position_error)
                pos_stds.append(acc_result.std_position_error)
                rot_means.append(acc_result.mean_rotation_error)
                rot_stds.append(acc_result.std_rotation_error)
            except AttributeError:
                pos_means.append(0.0)
                pos_stds.append(0.0)
                rot_means.append(0.0)
                rot_stds.append(0.0)
        
        # Position error bar plot
        x_pos = np.arange(len(methods))
        bars1 = ax1.bar(x_pos, pos_means, yerr=pos_stds, capsize=5, 
                       color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Position Error', fontsize=12, fontweight='bold')
        ax1.set_xlabel('IK Methods', fontsize=10)
        ax1.set_ylabel('Mean Position Error (m)', fontsize=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Rotation error bar plot  
        bars2 = ax2.bar(x_pos, rot_means, yerr=rot_stds, capsize=5,
                       color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Rotation Error', fontsize=12, fontweight='bold')
        ax2.set_xlabel('IK Methods', fontsize=10)
        ax2.set_ylabel('Mean Rotation Error (rad)', fontsize=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Accuracy comparison saved to {save_path}")
        
        return fig
    
    @staticmethod
    def create_summary_dashboard(all_results: Dict[str, Any],
                               title: str = "IK Benchmark Dashboard",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive summary dashboard.
        
        Args:
            all_results: Dict containing all benchmark results
            title: Dashboard title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Extract method names
        methods = list(all_results.keys()) if all_results else []
        
        if not methods:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No benchmark data available',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=20, color='red')
            return fig
        
        colors = sns.color_palette("husl", len(methods))
        
        # 1. Convergence rates (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        conv_rates = []
        for method in methods:
            try:
                rate = all_results[method].get('solvability', {}).get('convergence_rate', 0)
                conv_rates.append(rate)
            except:
                conv_rates.append(0)
        
        bars = ax1.bar(range(len(methods)), conv_rates, color=colors, alpha=0.7)
        ax1.set_title('Convergence Rate', fontweight='bold')
        ax1.set_ylabel('Rate')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # 2. Separated solve times (top center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Collect separated timing data
        overall_times = []
        success_times = []
        failed_times = []
        
        for method in methods:
            try:
                eff_data = all_results[method].get('efficiency', {})
                overall_times.append(eff_data.get('mean_solve_time', 0))
                success_times.append(eff_data.get('converged_mean_time', 0))
                failed_times.append(eff_data.get('failed_mean_time', 0))
            except:
                overall_times.append(0)
                success_times.append(0) 
                failed_times.append(0)
        
        # Create grouped bar chart for separated timing
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax2.bar(x - width, overall_times, width, label='Overall', color=colors[0] if colors else 'blue', alpha=0.8)
        bars2 = ax2.bar(x, success_times, width, label='Success', color=colors[1] if len(colors) > 1 else 'green', alpha=0.8)
        bars3 = ax2.bar(x + width, failed_times, width, label='Failed', color=colors[2] if len(colors) > 2 else 'red', alpha=0.8)
        
        ax2.set_title('Solve Time Breakdown', fontweight='bold')
        ax2.set_ylabel('Time (s)')
        ax2.set_xlabel('IK Methods')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45, ha='right')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        max_height = max(max(overall_times), max(success_times), max(failed_times)) if any(overall_times + success_times + failed_times) else 1
        for bar, time_val in zip(bars1, overall_times):
            if time_val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_height*0.01,
                        f'{time_val:.3f}', ha='center', va='bottom', fontsize=7)
        
        for bar, time_val in zip(bars2, success_times):
            if time_val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_height*0.01,
                        f'{time_val:.3f}', ha='center', va='bottom', fontsize=7)
                        
        for bar, time_val in zip(bars3, failed_times):
            if time_val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_height*0.01,
                        f'{time_val:.3f}', ha='center', va='bottom', fontsize=7)
        
        # 3. Position accuracy (top center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        pos_errors = []
        for method in methods:
            try:
                error = all_results[method].get('accuracy', {}).get('mean_position_error', 0)
                pos_errors.append(error if np.isfinite(error) else 0)
            except:
                pos_errors.append(0)
        
        bars = ax3.bar(range(len(methods)), pos_errors, color=colors, alpha=0.7)
        ax3.set_title('Position Accuracy', fontweight='bold')
        ax3.set_ylabel('Error (m)')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45, ha='right')
        
        # 4. Robustness score (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        robustness_scores = []
        for method in methods:
            try:
                # Compute composite robustness score
                rob_data = all_results[method].get('robustness', {})
                stability_metrics = rob_data.get('stability_metrics', {})
                initial_values = [stability_metrics.get(k, 0.0) for k in ['random', 'zero', 'middle', 'random_middle'] if k in stability_metrics]
                initial_avg = np.mean(initial_values) if initial_values else 0.0
                success_rates = rob_data.get('success_rates_by_noise', [])
                noise_avg = np.mean(success_rates) if success_rates else 0.0
                stability_metrics = rob_data.get('stability_metrics', {})
                continuity = stability_metrics.get('trajectory_continuity', 0.0)
                score = (initial_avg + noise_avg + continuity) / 3
                robustness_scores.append(score)
            except:
                robustness_scores.append(0)
        
        bars = ax4.bar(range(len(methods)), robustness_scores, color=colors, alpha=0.7)
        ax4.set_title('Robustness Score', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xticks(range(len(methods)))
        ax4.set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        
        # 5-6. Performance comparison table (middle row)
        ax5 = fig.add_subplot(gs[1, :2])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create performance table
        table_data = []
        headers = ['Method', 'Conv Rate', 'Overall Time (s)', 'Success Time (s)', 'Failed Time (s)', 'Pos Error (m)', 'Rob Score']
        
        for i, method in enumerate(methods):
            row = [
                method.replace('_', ' ').title(),
                f"{conv_rates[i]:.2%}",
                f"{overall_times[i]:.4f}",
                f"{success_times[i]:.4f}",
                f"{failed_times[i]:.4f}",
                f"{pos_errors[i]:.2e}" if pos_errors[i] > 0 else "0.00e+00",
                f"{robustness_scores[i]:.3f}"
            ]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colColours=['lightgray'] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Smaller font for more columns
        table.scale(1.4, 1.3)  # Wider to accommodate new columns
        ax5.set_title('Performance Summary Table', fontweight='bold', pad=20)
        
        # 6. Notes/recommendations (middle right)
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')
        
        # Generate recommendations based on results
        if methods:
            best_conv = methods[np.argmax(conv_rates)] if conv_rates else "N/A"
            best_speed = methods[np.argmin([t if t > 0 else float('inf') for t in overall_times])] if overall_times else "N/A"
            best_accuracy = methods[np.argmin([e if e > 0 else float('inf') for e in pos_errors])] if pos_errors else "N/A"
            best_robust = methods[np.argmax(robustness_scores)] if robustness_scores else "N/A"
            
            recommendations = f"""
BENCHMARK SUMMARY & RECOMMENDATIONS

Best Convergence Rate: {best_conv.replace('_', ' ').title()}
Best Speed: {best_speed.replace('_', ' ').title()}  
Best Accuracy: {best_accuracy.replace('_', ' ').title()}
Best Robustness: {best_robust.replace('_', ' ').title()}

RECOMMENDATIONS:
• For high success rates: Use {best_conv}
• For real-time applications: Use {best_speed}
• For precision tasks: Use {best_accuracy}
• For varying conditions: Use {best_robust}
            """
        else:
            recommendations = "No data available for recommendations."
        
        ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 7. Overall ranking (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Compute overall scores (normalized)
        if methods:
            overall_scores = []
            for i, method in enumerate(methods):
                # Normalize metrics to 0-1 scale and compute weighted average
                norm_conv = conv_rates[i] 
                norm_speed = 1 - (overall_times[i] / max(overall_times)) if max(overall_times) > 0 else 0
                norm_acc = 1 - (pos_errors[i] / max(pos_errors)) if max(pos_errors) > 0 else 1
                norm_rob = robustness_scores[i]
                
                # Weighted overall score (adjust weights as needed)
                overall = 0.3*norm_conv + 0.2*norm_speed + 0.3*norm_acc + 0.2*norm_rob
                overall_scores.append(overall)
            
            # Sort methods by overall score
            sorted_methods = sorted(zip(methods, overall_scores), key=lambda x: x[1], reverse=True)
            
            ranking_text = "OVERALL RANKING (Weighted Score):\n\n"
            for rank, (method, score) in enumerate(sorted_methods, 1):
                ranking_text += f"{rank}. {method.replace('_', ' ').title():<20} Score: {score:.3f}\n"
            
            ax7.text(0.5, 0.5, ranking_text, transform=ax7.transAxes,
                    fontsize=12, ha='center', va='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for main title
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Summary dashboard saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_3d_pose_scatter(pose_data: Dict[str, List[Tuple[np.ndarray, bool]]], 
                            title: str = "3D Pose Success/Failure Distribution",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D scatter visualization of pose solve success/failure distribution.
        
        Shows target poses in 3D space color-coded by success (green) or failure (red)
        for each IK method, similar to scatter RGB frames visualization.
        
        Args:
            pose_data: Dict mapping method names to list of (target_pose_4x4, converged) tuples
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object with 3D subplots
        """
        if not pose_data:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No pose data available for 3D visualization',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, color='red')
            return fig
        
        # Determine subplot layout
        n_methods = len(pose_data)
        n_cols = min(3, n_methods)  # Max 3 columns for readability
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        
        for i, (method_name, pose_results) in enumerate(pose_data.items()):
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
            
            if not pose_results:
                ax.text(0.5, 0.5, 0.5, f'No data for {method_name}',
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            # Extract positions and success flags
            positions = []
            success_flags = []
            
            for target_pose, converged in pose_results:
                if target_pose is not None and target_pose.shape == (4, 4):
                    positions.append(target_pose[:3, 3])  # Extract translation
                    success_flags.append(converged)
            
            if not positions:
                ax.text(0.5, 0.5, 0.5, f'Invalid pose data for {method_name}',
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            positions = np.array(positions)
            success_flags = np.array(success_flags)
            
            # Limit points for performance (max 500 per method)
            if len(positions) > 500:
                indices = np.random.choice(len(positions), 500, replace=False)
                positions = positions[indices]
                success_flags = success_flags[indices]
            
            # Separate successful and failed poses
            success_pos = positions[success_flags]
            failed_pos = positions[~success_flags]
            
            # Plot successful poses (green)
            if len(success_pos) > 0:
                ax.scatter(success_pos[:, 0], success_pos[:, 1], success_pos[:, 2],
                          c='green', marker='o', s=20, alpha=0.7, 
                          label=f'Success ({len(success_pos)})')
            
            # Plot failed poses (red)
            if len(failed_pos) > 0:
                ax.scatter(failed_pos[:, 0], failed_pos[:, 1], failed_pos[:, 2],
                          c='red', marker='x', s=25, alpha=0.8,
                          label=f'Failed ({len(failed_pos)})')
            
            # Customize subplot
            ax.set_title(f'{method_name.replace("_", " ").title()}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('X Position (m)', fontsize=9)
            ax.set_ylabel('Y Position (m)', fontsize=9)
            ax.set_zlabel('Z Position (m)', fontsize=9)
            
            # Add legend
            if len(success_pos) > 0 or len(failed_pos) > 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # Set equal aspect ratio for better visualization
            max_range = 0.1
            if len(positions) > 0:
                max_range = np.max(np.abs(positions)) * 1.1
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"3D pose scatter plot saved to {save_path}")
        
        return fig
    
    @staticmethod  
    def plot_workspace_3d_coverage(workspace_points: np.ndarray,
                                  success_mask: np.ndarray,
                                  method_name: str,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D workspace coverage visualization for a single IK method.
        
        Args:
            workspace_points: Position points array (N, 3)
            success_mask: Boolean success mask array (N,)
            method_name: IK solver name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if workspace_points.size == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No workspace data available',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Separate successful and failed points
        success_points = workspace_points[success_mask]
        failed_points = workspace_points[~success_mask]
        
        # Plot successful points
        if len(success_points) > 0:
            ax.scatter(success_points[:, 0], success_points[:, 1], success_points[:, 2],
                      c='green', marker='o', s=15, alpha=0.6,
                      label=f'Reachable ({len(success_points)})')
        
        # Plot failed points
        if len(failed_points) > 0:
            ax.scatter(failed_points[:, 0], failed_points[:, 1], failed_points[:, 2],
                      c='red', marker='x', s=20, alpha=0.4,
                      label=f'Unreachable ({len(failed_points)})')
        
        # Customize plot
        ax.set_title(f'3D Workspace Coverage - {method_name.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        
        # Add legend and statistics
        total_points = len(workspace_points)
        success_rate = len(success_points) / total_points if total_points > 0 else 0.0
        ax.legend(title=f'Success Rate: {success_rate:.1%}', loc='upper left')
        
        # Set reasonable axis limits
        if workspace_points.size > 0:
            margin = 0.05
            x_range = workspace_points[:, 0].max() - workspace_points[:, 0].min()
            y_range = workspace_points[:, 1].max() - workspace_points[:, 1].min()
            z_range = workspace_points[:, 2].max() - workspace_points[:, 2].min()
            
            ax.set_xlim(workspace_points[:, 0].min() - margin * x_range,
                       workspace_points[:, 0].max() + margin * x_range)
            ax.set_ylim(workspace_points[:, 1].min() - margin * y_range,
                       workspace_points[:, 1].max() + margin * y_range) 
            ax.set_zlim(workspace_points[:, 2].min() - margin * z_range,
                       workspace_points[:, 2].max() + margin * z_range)
        
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"3D workspace coverage saved to {save_path}")
        
        return fig