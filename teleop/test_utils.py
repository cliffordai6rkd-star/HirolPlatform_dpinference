import pytest
from utils import RisingEdgeDetector


class TestRisingEdgeDetector:
    """Test suite for RisingEdgeDetector class."""
    
    def test_init_default_threshold(self):
        """Test initialization with default threshold."""
        detector = RisingEdgeDetector()
        assert detector._threshold == 0.5
        assert detector._previous_value == 0.0
    
    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        detector = RisingEdgeDetector(threshold=1.0)
        assert detector._threshold == 1.0
        assert detector._previous_value == 0.0
    
    def test_init_invalid_threshold(self):
        """Test initialization with invalid threshold."""
        with pytest.raises(AssertionError):
            RisingEdgeDetector(threshold="invalid")
    
    def test_basic_rising_edge_detection(self):
        """Test basic rising edge detection from low to high."""
        detector = RisingEdgeDetector(threshold=0.5)
        
        # Initial low signal - no rising edge
        assert detector.update(0.0) == False
        
        # Signal goes high - rising edge detected
        assert detector.update(1.0) == True
        
        # Signal stays high - no rising edge
        assert detector.update(1.0) == False
        
        # Signal goes low - no rising edge
        assert detector.update(0.0) == False
        
        # Signal goes high again - rising edge detected
        assert detector.update(1.0) == True
    
    def test_threshold_boundary_conditions(self):
        """Test behavior at threshold boundary."""
        detector = RisingEdgeDetector(threshold=0.5)
        
        # Start below threshold
        assert detector.update(0.5) == False  # exactly at threshold
        assert detector.update(0.51) == True  # just above threshold - rising edge
        assert detector.update(0.5) == False  # back to threshold - falling edge
        assert detector.update(0.49) == False  # below threshold
        assert detector.update(0.5) == False  # back to threshold
        assert detector.update(0.51) == True  # above threshold - rising edge
    
    def test_negative_values(self):
        """Test with negative signal values."""
        detector = RisingEdgeDetector(threshold=0.0)
        
        assert detector.update(-1.0) == False  # below threshold
        assert detector.update(0.1) == True   # above threshold - rising edge
        assert detector.update(-0.5) == False  # below threshold
        assert detector.update(0.5) == True   # above threshold - rising edge
    
    def test_large_values(self):
        """Test with large signal values."""
        detector = RisingEdgeDetector(threshold=100.0)
        
        assert detector.update(50.0) == False   # below threshold
        assert detector.update(150.0) == True   # above threshold - rising edge
        assert detector.update(200.0) == False  # stays above threshold
        assert detector.update(50.0) == False   # below threshold
        assert detector.update(101.0) == True   # above threshold - rising edge
    
    def test_integer_signals(self):
        """Test with integer signal values."""
        detector = RisingEdgeDetector(threshold=2)
        
        assert detector.update(1) == False  # below threshold
        assert detector.update(3) == True   # above threshold - rising edge
        assert detector.update(5) == False  # stays above threshold
        assert detector.update(1) == False  # below threshold
        assert detector.update(4) == True   # above threshold - rising edge
    
    def test_invalid_signal_type(self):
        """Test with invalid signal types."""
        detector = RisingEdgeDetector()
        
        with pytest.raises(AssertionError):
            detector.update("invalid")
        
        with pytest.raises(AssertionError):
            detector.update(None)
        
        with pytest.raises(AssertionError):
            detector.update([1, 2, 3])
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        detector = RisingEdgeDetector(threshold=0.5)
        
        # Set up some state
        detector.update(1.0)  # high signal
        assert detector._previous_value == 1.0
        
        # Reset the detector
        detector.reset()
        assert detector._previous_value == 0.0
        
        # After reset, next high signal should trigger rising edge
        assert detector.update(1.0) == True
    
    def test_continuous_low_signals(self):
        """Test continuous low signals don't trigger rising edge."""
        detector = RisingEdgeDetector(threshold=0.5)
        
        for _ in range(10):
            assert detector.update(0.0) == False
    
    def test_continuous_high_signals(self):
        """Test continuous high signals only trigger once."""
        detector = RisingEdgeDetector(threshold=0.5)
        
        # First high signal should trigger rising edge
        assert detector.update(1.0) == True
        
        # Subsequent high signals should not trigger
        for _ in range(10):
            assert detector.update(1.0) == False
    
    def test_alternating_signals(self):
        """Test alternating high/low signals."""
        detector = RisingEdgeDetector(threshold=0.5)
        
        expected_results = [False, True, False, True, False, True, False]
        signals = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        
        for signal, expected in zip(signals, expected_results):
            assert detector.update(signal) == expected