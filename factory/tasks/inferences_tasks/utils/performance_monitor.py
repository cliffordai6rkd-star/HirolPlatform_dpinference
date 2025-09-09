#!/usr/bin/env python3
"""
Performance Monitor for ACT Inference
Handles performance profiling and statistics reporting
"""

import glog as log
from typing import Dict, Any
from tools.performance_profiler import PerformanceProfiler


class PerformanceMonitor:
    """Handles performance monitoring and reporting for ACT inference"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize performance monitor

        Args:
            config: Configuration dictionary
        """
        # Performance profiling configuration
        self.performance_config = config.get("performance_profiling", {})
        self.enable_performance_stats = self.performance_config.get("enabled", True)
        self.performance_print_threshold = self.performance_config.get("print_threshold_ms", 0.1)
        self.performance_detailed_report = self.performance_config.get("detailed_report", True)

        # Configure performance profiler
        if self.enable_performance_stats:
            PerformanceProfiler.enable(True)
            PerformanceProfiler.set_print_threshold(self.performance_print_threshold)
            log.info(f"📊 Performance profiling enabled (threshold: {self.performance_print_threshold}ms)")
        else:
            PerformanceProfiler.enable(False)
            log.info("📊 Performance profiling disabled")

    def print_episode_performance_stats(self, episode_num: int, step_count: int) -> None:
        """Print performance statistics for the completed episode"""
        if not self.enable_performance_stats:
            return

        log.info("=" * 60)
        log.info(f"📊 EPISODE {episode_num} PERFORMANCE SUMMARY")
        log.info("=" * 60)

        # Get ACT inference specific stats
        act_stats = PerformanceProfiler.get_stats()

        # Filter for ACT-related operations
        act_operations = {
            name: stats for name, stats in act_stats.items()
            if 'act_inference' in name
        }

        if act_operations:
            # Calculate total inference time
            total_inference_time = sum(
                stats['total_ms'] for name, stats in act_operations.items()
                if 'act_inference_time' in name
            )

            # Calculate average times per step
            avg_times = {}
            for name, stats in act_operations.items():
                key = name.replace('act_inference', '').strip()
                if key:
                    avg_times[key] = stats['avg_ms']

            log.info(f"Episode Steps: {step_count}")
            log.info(f"Total Inference Time: {total_inference_time:.2f}ms")
            if step_count > 0:
                log.info(f"Average Time per Step: {total_inference_time/step_count:.2f}ms")

            # Print key timing breakdowns
            key_timings = [
                ('model_inference', 'Model Inference'),
                ('obs_conversion_total', 'Observation Conversion'),
                ('action_conversion_total', 'Action Conversion'),
                ('robot_step_execution', 'Robot Execution'),
                ('gripper_control_processing', 'Gripper Control')
            ]

            log.info("\nKey Timing Breakdowns:")
            for key, label in key_timings:
                if key in avg_times:
                    log.info(f"  {label}: {avg_times[key]:.2f}ms avg")

            # Check for slow operations
            slow_ops = [
                (name, stats['avg_ms']) for name, stats in act_operations.items()
                if stats['avg_ms'] > 10.0  # > 10ms
            ]

            if slow_ops:
                log.info("\n⚠️  Slow Operations (>10ms):")
                for name, avg_ms in sorted(slow_ops, key=lambda x: x[1], reverse=True)[:3]:
                    log.info(f"  {name}: {avg_ms:.2f}ms")

        log.info("=" * 60)

    def print_final_performance_report(self) -> None:
        """Print comprehensive performance report at the end of all episodes"""
        if not self.enable_performance_stats:
            return

        log.info("=" * 80)
        log.info("📊 FINAL PERFORMANCE REPORT")
        log.info("=" * 80)

        # Print summary
        PerformanceProfiler.print_summary()

        # Print detailed stats if enabled
        if self.performance_detailed_report:
            log.info("\nDetailed Performance Statistics:")
            PerformanceProfiler.print_stats(
                sort_by='avg_ms',
                top_n=15,
                detailed=True
            )

        # Get and display ACT-specific insights
        act_stats = PerformanceProfiler.get_stats()
        act_operations = {
            name: stats for name, stats in act_stats.items()
            if 'act_inference' in name
        }

        if act_operations:
            log.info("\n🤖 ACT INFERENCE INSIGHTS:")

            # Model inference analysis
            model_stats = act_stats.get('act_inferencemodel_inference', {})
            if model_stats:
                log.info(f"  Model Inference: {model_stats['avg_ms']:.2f}ms avg, {model_stats['count']} calls")
                if model_stats['avg_ms'] > 50:
                    log.warning("  ⚠️  Model inference is slow (>50ms) - consider optimizing")

            # Data conversion analysis
            obs_stats = act_stats.get('act_inferenceobs_conversion_total', {})
            if obs_stats:
                log.info(f"  Observation Conversion: {obs_stats['avg_ms']:.2f}ms avg")

            action_stats = act_stats.get('act_inferenceaction_conversion_total', {})
            if action_stats:
                log.info(f"  Action Conversion: {action_stats['avg_ms']:.2f}ms avg")

            # Frequency analysis
            total_calls = sum(stats['count'] for stats in act_operations.values())
            total_time_s = sum(stats['total_ms'] for stats in act_operations.values()) / 1000
            if total_time_s > 0:
                effective_hz = total_calls / total_time_s
                log.info(f"  Effective Processing Rate: {effective_hz:.1f} Hz")

        log.info("=" * 80)

    def is_enabled(self) -> bool:
        """Check if performance monitoring is enabled"""
        return self.enable_performance_stats