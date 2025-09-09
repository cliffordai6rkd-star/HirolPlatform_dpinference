#!/usr/bin/env python3
"""
Keyboard Handler for ACT Inference
Handles keyboard input and user interaction
"""

import time
import threading
import glog as log
from typing import List, Optional
from pynput import keyboard


class KeyboardHandler:
    """Handles keyboard input and user interaction for ACT inference"""

    def __init__(self) -> None:
        """Initialize keyboard handler"""
        self.key_pressed: Optional[str] = None
        self.keyboard_listener: Optional[keyboard.Listener] = None
        self.waiting_for_key = False
        self.key_event = threading.Event()

        log.info("✅ Keyboard handler initialized")

    def on_key_press(self, key) -> None:
        """Handle key press events"""
        if hasattr(key, 'char') and key.char:
            self.key_pressed = key.char
        elif key == keyboard.Key.esc:
            self.key_pressed = 'q'
        else:
            # Handle special keys
            return

        if self.waiting_for_key:
            self.key_event.set()

    def start_keyboard_listener(self) -> None:
        """Start the keyboard listener"""
        if self.keyboard_listener is None or not self.keyboard_listener.running:
            self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
            self.keyboard_listener.start()

    def stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener"""
        if self.keyboard_listener and self.keyboard_listener.running:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

    def wait_for_key_press(self, prompt_message: str, valid_keys: List[str] = None) -> str:
        """
        Wait for a specific key press using pynput

        Args:
            prompt_message: Message to display to user
            valid_keys: List of valid keys to accept

        Returns:
            The pressed key
        """
        if valid_keys is None:
            valid_keys = ['s', 'q']

        self.start_keyboard_listener()

        while True:
            print(prompt_message)
            self.waiting_for_key = True
            self.key_event.clear()

            # Wait for key press event or timeout after 0.5 seconds
            if self.key_event.wait(timeout=0.5):
                self.waiting_for_key = False
                if self.key_pressed in valid_keys:
                    return self.key_pressed
                else:
                    print(f"Invalid key '{self.key_pressed}'. Please press one of: {valid_keys}")
            else:
                # Timeout occurred, continue the loop to show prompt again
                self.waiting_for_key = False

            time.sleep(0.05)

    def cleanup(self) -> None:
        """Clean up keyboard handler resources"""
        self.stop_keyboard_listener()