#!/usr/bin/env python3
"""
Auto-run version of AgiBot G1 periodic motion test.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from test_periodic_motion import run_periodic_motion, AgibotG1
import glog as log
import numpy as np


def main():
    """Run periodic motion test automatically."""
    # Force mock mode
    os.environ['AGIBOT_USE_MOCK'] = '1'
    
    # Robot configuration
    config = {
        'dof': [7, 7],  # Dual 7-DOF arms
        'robot_name': 'AgiBot_G1_Demo',
        'control_head': False,
        'control_waist': False,
        'control_wheel': False,
        'control_gripper': False,
        'control_hand': False,
    }
    
    log.info("=" * 60)
    log.info("AgiBot G1 Automatic Periodic Motion Demo")
    log.info("=" * 60)
    
    try:
        # Initialize robot
        log.info("Initializing AgiBot G1...")
        robot = AgibotG1(config)
        
        if not robot._is_initialized:
            log.error("Robot initialization failed")
            return 1
        
        log.info(f"Robot initialized with {robot._total_dof} DOF")
        log.info(f"Using {'mock' if robot._use_mock else 'real'} robot implementation")
        
        # Run different motion patterns
        patterns = [
            {"name": "Slow sine wave", "frequency": 0.3, "amplitude": 8, "duration": 5},
            {"name": "Medium frequency", "frequency": 0.8, "amplitude": 12, "duration": 4},
            {"name": "Fast small motion", "frequency": 1.5, "amplitude": 5, "duration": 3},
        ]
        
        for i, pattern in enumerate(patterns):
            log.info(f"\n--- Pattern {i+1}: {pattern['name']} ---")
            run_periodic_motion(
                robot,
                frequency=pattern['frequency'],
                amplitude_deg=pattern['amplitude'],
                duration=pattern['duration'],
                control_rate=50.0
            )
            
            if i < len(patterns) - 1:
                log.info("Pausing for 2 seconds...")
                import time
                time.sleep(2.0)
        
        log.info("\nAll motion patterns completed successfully!")
        
    except Exception as e:
        log.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        if 'robot' in locals():
            log.info("Shutting down robot...")
            robot.close()
            
        # Clean up environment
        if 'AGIBOT_USE_MOCK' in os.environ:
            del os.environ['AGIBOT_USE_MOCK']
    
    return 0


if __name__ == "__main__":
    sys.exit(main())