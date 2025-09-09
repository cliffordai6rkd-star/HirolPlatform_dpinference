from typing import Union

class RisingEdgeDetector:
    """
    A detector for rising edge transitions in signals.
    
    Detects transitions from low level (below threshold) to high level (above threshold).
    """
    
    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the rising edge detector.
        
        Args:
            threshold: The threshold value to distinguish between high and low levels.
                      Default is 0.5.
        
        Raises:
            AssertionError: If threshold is not a valid number.
        """
        assert isinstance(threshold, (int, float)), "Threshold must be a number"
        self._threshold: float = float(threshold)
        self._previous_value: float = 0.0
    
    def update(self, signal: Union[int, float]) -> bool:
        """
        Update the detector with a new signal value and check for rising edge.
        
        A rising edge is detected when:
        - Current signal > threshold AND
        - Previous signal <= threshold
        
        Args:
            signal: The current signal value to process.
            
        Returns:
            bool: True if a rising edge is detected, False otherwise.
            
        Raises:
            AssertionError: If signal is not a valid number.
        """
        assert isinstance(signal, (int, float)), "Signal must be a number"
        
        current_high = signal > self._threshold
        previous_high = self._previous_value > self._threshold
        
        rising_edge = current_high and not previous_high
        
        self._previous_value = float(signal)
        
        return rising_edge
    
    def reset(self) -> None:
        """
        Reset the detector state.
        
        Sets the previous value to 0.0, effectively treating the next signal
        as if it's the first signal received.
        """
        self._previous_value = 0.0
        