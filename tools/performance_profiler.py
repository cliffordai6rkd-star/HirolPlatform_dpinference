#!/usr/bin/env python3
"""
Performance Profiler - Thread-safe timing utility for code optimization

Features:
- Thread-safe timing statistics
- Context manager support  
- Automatic averaging and statistics
- Configurable output formats
- Minimal overhead design

Usage:
    # Method 1: Context manager (recommended)
    with PerformanceProfiler.timer("my_function"):
        # Your code here
        pass
    
    # Method 2: Manual timing
    profiler = PerformanceProfiler()
    profiler.start("operation_name")
    # Your code here
    profiler.end("operation_name")
    
    # Method 3: Decorator
    @PerformanceProfiler.profile("function_name")
    def my_function():
        pass
    
    # Print statistics
    PerformanceProfiler.print_stats()
"""

import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics
import glog as log


class PerformanceProfiler:
    """Thread-safe performance profiler with statistical analysis"""
    
    # Class-level shared data structures with locks
    _lock = threading.RLock()
    _timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 measurements
    _active_timers: Dict[str, float] = {}  # Per-thread active timers
    _call_counts: Dict[str, int] = defaultdict(int)
    _enabled = True
    _print_threshold_ms = 0.1  # Only print operations > 0.1ms by default
    
    def __init__(self, name_prefix: str = ""):
        """Initialize profiler instance
        
        Args:
            name_prefix: Prefix for all timing names from this instance
        """
        self.name_prefix = name_prefix
    
    @classmethod
    def enable(cls, enabled: bool = True):
        """Enable or disable profiling globally"""
        with cls._lock:
            cls._enabled = enabled
    
    @classmethod  
    def set_print_threshold(cls, threshold_ms: float):
        """Set minimum time (ms) to print in statistics"""
        with cls._lock:
            cls._print_threshold_ms = threshold_ms
    
    @classmethod
    def clear_stats(cls):
        """Clear all timing statistics"""
        with cls._lock:
            cls._timings.clear()
            cls._call_counts.clear()
            cls._active_timers.clear()
    
    def _get_thread_key(self, name: str) -> str:
        """Get thread-specific key for timing"""
        thread_id = threading.current_thread().ident
        return f"{self.name_prefix}{name}_{thread_id}"
    
    def start(self, name: str) -> None:
        """Start timing an operation
        
        Args:
            name: Name of the operation to time
        """
        if not self._enabled:
            return
            
        thread_key = self._get_thread_key(name)
        with self._lock:
            self._active_timers[thread_key] = time.perf_counter()
    
    def end(self, name: str) -> Optional[float]:
        """End timing an operation
        
        Args:
            name: Name of the operation to time
            
        Returns:
            Duration in seconds, or None if not started
        """
        if not self._enabled:
            return None
            
        end_time = time.perf_counter()
        thread_key = self._get_thread_key(name)
        
        with self._lock:
            if thread_key not in self._active_timers:
                log.warning(f"Timer '{name}' was not started")
                return None
                
            start_time = self._active_timers.pop(thread_key)
            duration = end_time - start_time
            
            # Store timing data
            display_name = f"{self.name_prefix}{name}"
            self._timings[display_name].append(duration)
            self._call_counts[display_name] += 1
            
            return duration
    
    @classmethod
    @contextmanager
    def timer(cls, name: str, instance_prefix: str = ""):
        """Context manager for timing code blocks
        
        Args:
            name: Name of the operation
            instance_prefix: Optional prefix for this timing
            
        Example:
            with PerformanceProfiler.timer("database_query"):
                # Your code here
                pass
        """
        if not cls._enabled:
            yield None
            return
            
        profiler = cls(instance_prefix)
        profiler.start(name)
        try:
            yield profiler
        finally:
            profiler.end(name)
    
    @classmethod
    def profile(cls, name: str, instance_prefix: str = ""):
        """Decorator for timing functions
        
        Args:
            name: Name of the operation
            instance_prefix: Optional prefix for this timing
            
        Example:
            @PerformanceProfiler.profile("my_function")
            def my_function():
                pass
        """
        def decorator(func):
            if not cls._enabled:
                return func
                
            def wrapper(*args, **kwargs):
                with cls.timer(name, instance_prefix):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @classmethod
    def get_stats(cls, name: str = None) -> Dict[str, Any]:
        """Get timing statistics
        
        Args:
            name: Specific operation name, or None for all
            
        Returns:
            Dictionary of statistics
        """
        with cls._lock:
            if name and name not in cls._timings:
                return {}
            
            names_to_process = [name] if name else list(cls._timings.keys())
            stats = {}
            
            for op_name in names_to_process:
                timings = list(cls._timings[op_name])
                if not timings:
                    continue
                
                # Convert to milliseconds for readability
                timings_ms = [t * 1000 for t in timings]
                
                stats[op_name] = {
                    'count': cls._call_counts[op_name],
                    'total_ms': sum(timings_ms),
                    'avg_ms': statistics.mean(timings_ms),
                    'min_ms': min(timings_ms),
                    'max_ms': max(timings_ms),
                    'median_ms': statistics.median(timings_ms),
                    'std_dev_ms': statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
                    'recent_samples': len(timings)
                }
            
            return stats
    
    @classmethod
    def print_stats(cls, name: str = None, sort_by: str = 'total_ms', 
                   top_n: int = None, detailed: bool = False):
        """Print timing statistics
        
        Args:
            name: Specific operation name, or None for all
            sort_by: Sort key ('total_ms', 'avg_ms', 'count', 'max_ms')
            top_n: Show only top N results
            detailed: Include detailed statistics
        """
        stats = cls.get_stats(name)
        if not stats:
            log.info("No timing statistics available")
            return
        
        # Filter by threshold
        filtered_stats = {
            k: v for k, v in stats.items() 
            if v['avg_ms'] >= cls._print_threshold_ms
        }
        
        if not filtered_stats:
            log.info(f"No operations above {cls._print_threshold_ms:.1f}ms threshold")
            return
        
        # Sort results
        sorted_stats = sorted(filtered_stats.items(), 
                            key=lambda x: x[1][sort_by], 
                            reverse=True)
        
        if top_n:
            sorted_stats = sorted_stats[:top_n]
        
        # Print header
        log.info("=" * 80)
        log.info(f"PERFORMANCE STATISTICS (sorted by {sort_by}, threshold: {cls._print_threshold_ms:.1f}ms)")
        log.info("=" * 80)
        
        # Print table header
        if detailed:
            log.info(f"{'Operation':<30} {'Count':<8} {'Total':<10} {'Avg':<10} {'Min':<8} {'Max':<10} {'Median':<8} {'StdDev':<8}")
            log.info("-" * 104)
        else:
            log.info(f"{'Operation':<40} {'Count':<8} {'Total':<12} {'Avg':<12}")
            log.info("-" * 80)
        
        # Print data
        for op_name, data in sorted_stats:
            if detailed:
                log.info(f"{op_name:<30} {data['count']:<8} "
                        f"{data['total_ms']:<10.2f} {data['avg_ms']:<10.2f} "
                        f"{data['min_ms']:<8.2f} {data['max_ms']:<10.2f} "
                        f"{data['median_ms']:<8.2f} {data['std_dev_ms']:<8.2f}")
            else:
                log.info(f"{op_name:<40} {data['count']:<8} "
                        f"{data['total_ms']:<12.2f} {data['avg_ms']:<12.2f}")
        
        log.info("=" * 80)
    
    @classmethod
    def get_slow_operations(cls, threshold_ms: float = 10.0) -> List[tuple]:
        """Get operations slower than threshold
        
        Args:
            threshold_ms: Threshold in milliseconds
            
        Returns:
            List of (name, avg_time_ms) tuples
        """
        stats = cls.get_stats()
        slow_ops = [
            (name, data['avg_ms']) 
            for name, data in stats.items() 
            if data['avg_ms'] > threshold_ms
        ]
        return sorted(slow_ops, key=lambda x: x[1], reverse=True)
    
    @classmethod
    def print_summary(cls):
        """Print a brief summary of all timings"""
        stats = cls.get_stats()
        if not stats:
            log.info("No performance data collected")
            return
        
        total_ops = sum(data['count'] for data in stats.values())
        total_time_ms = sum(data['total_ms'] for data in stats.values())
        
        slow_ops = cls.get_slow_operations(5.0)  # > 5ms
        
        log.info("=" * 50)
        log.info("PERFORMANCE SUMMARY")
        log.info("=" * 50)
        log.info(f"Total operations tracked: {len(stats)}")
        log.info(f"Total function calls: {total_ops}")
        log.info(f"Total time spent: {total_time_ms:.2f}ms")
        
        if slow_ops:
            log.info(f"\nSlowest operations (>5ms):")
            for name, avg_ms in slow_ops[:5]:  # Top 5 slowest
                log.info(f"  {name}: {avg_ms:.2f}ms avg")
        
        log.info("=" * 50)


# Convenience aliases for common usage
perf = PerformanceProfiler
timer = PerformanceProfiler.timer
profile = PerformanceProfiler.profile


if __name__ == "__main__":
    # Example usage and testing
    import random
    
    # Test basic functionality
    profiler = PerformanceProfiler("test_")
    
    # Test manual timing
    for i in range(10):
        profiler.start("manual_test")
        time.sleep(random.uniform(0.001, 0.01))
        profiler.end("manual_test")
    
    # Test context manager
    for i in range(5):
        with PerformanceProfiler.timer("context_test"):
            time.sleep(random.uniform(0.002, 0.005))
    
    # Test decorator
    @PerformanceProfiler.profile("decorated_function")
    def test_function():
        time.sleep(random.uniform(0.001, 0.003))
    
    for i in range(8):
        test_function()
    
    # Print results
    PerformanceProfiler.print_summary()
    PerformanceProfiler.print_stats(detailed=True)