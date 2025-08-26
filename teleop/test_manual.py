#!/usr/bin/env python3
"""Manual test script for RisingEdgeDetector without pytest dependency."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils import RisingEdgeDetector


def test_basic_functionality():
    """Test basic rising edge detection."""
    print("Testing basic functionality...")
    detector = RisingEdgeDetector(threshold=0.5)
    
    # Test sequence: low -> high -> high -> low -> high
    test_cases = [
        (0.0, False, "Initial low signal"),
        (1.0, True, "Signal goes high - rising edge"),
        (1.0, False, "Signal stays high - no rising edge"),
        (0.0, False, "Signal goes low - no rising edge"),
        (1.0, True, "Signal goes high again - rising edge")
    ]
    
    for signal, expected, description in test_cases:
        result = detector.update(signal)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {description}: update({signal}) -> {result} (expected {expected})")
        assert result == expected, f"Failed: {description}"
    
    print("Basic functionality test passed!")


def test_threshold_boundary():
    """Test threshold boundary conditions."""
    print("Testing threshold boundary...")
    detector = RisingEdgeDetector(threshold=0.5)
    
    test_cases = [
        (0.5, False, "At threshold"),
        (0.51, True, "Just above threshold - rising edge"),
        (0.5, False, "Back to threshold"),
        (0.49, False, "Below threshold"),
        (0.51, True, "Above threshold again - rising edge")
    ]
    
    for signal, expected, description in test_cases:
        result = detector.update(signal)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {description}: update({signal}) -> {result} (expected {expected})")
        assert result == expected, f"Failed: {description}"
    
    print("Threshold boundary test passed!")


def test_reset_functionality():
    """Test reset functionality."""
    print("Testing reset functionality...")
    detector = RisingEdgeDetector(threshold=0.5)
    
    # Set up some state
    detector.update(1.0)
    print(f"  After update(1.0), previous_value = {detector._previous_value}")
    
    # Reset
    detector.reset()
    print(f"  After reset(), previous_value = {detector._previous_value}")
    assert detector._previous_value == 0.0, "Reset failed"
    
    # After reset, next high signal should trigger rising edge
    result = detector.update(1.0)
    print(f"  After reset, update(1.0) -> {result}")
    assert result == True, "Rising edge not detected after reset"
    
    print("Reset functionality test passed!")


def test_invalid_inputs():
    """Test invalid input handling."""
    print("Testing invalid input handling...")
    
    # Test invalid threshold
    try:
        RisingEdgeDetector(threshold="invalid")
        print("  ✗ Should have raised AssertionError for invalid threshold")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print("  ✓ Correctly rejected invalid threshold")
    
    # Test invalid signal
    detector = RisingEdgeDetector()
    try:
        detector.update("invalid")
        print("  ✗ Should have raised AssertionError for invalid signal")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print("  ✓ Correctly rejected invalid signal")
    
    print("Invalid input handling test passed!")


def test_edge_cases():
    """Test edge cases with different value ranges."""
    print("Testing edge cases...")
    
    # Negative values
    detector = RisingEdgeDetector(threshold=0.0)
    assert detector.update(-1.0) == False, "Failed negative value test"
    assert detector.update(0.1) == True, "Failed negative to positive transition"
    print("  ✓ Negative values test passed")
    
    # Large values
    detector = RisingEdgeDetector(threshold=100.0)
    assert detector.update(50.0) == False, "Failed large values test (below)"
    assert detector.update(150.0) == True, "Failed large values test (above)"
    assert detector.update(200.0) == False, "Failed large values test (stay above)"
    print("  ✓ Large values test passed")
    
    # Integer signals
    detector = RisingEdgeDetector(threshold=2)
    assert detector.update(1) == False, "Failed integer test (below)"
    assert detector.update(3) == True, "Failed integer test (above)"
    print("  ✓ Integer signals test passed")
    
    print("Edge cases test passed!")


def main():
    """Run all tests."""
    print("Running RisingEdgeDetector tests...\n")
    
    try:
        test_basic_functionality()
        print()
        test_threshold_boundary()
        print()
        test_reset_functionality()
        print()
        test_invalid_inputs()
        print()
        test_edge_cases()
        print()
        
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)