import pygame
import re

class Controller:
    def __init__(self, controller_type=None):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.running = False
        self.controller_type = controller_type
        self.detected_type = None
        
        # Track previous button and axis states to detect changes
        self.prev_buttons = {}
        self.prev_axes = {}
        self.prev_hats = {}
        
        # Define controller mappings
        self._setup_controller_mappings()
        
        # Will be set based on detected/specified controller type
        self.button_map = {}
        self.axis_map = {}
        self.dpad_map = {}
    
    def _setup_controller_mappings(self):
        """Setup button and axis mappings for different controller types"""
        
        # PS5/PS4 controller mappings
        self.ps5_button_map = {
            0: "X",
            1: "Circle", 
            2: "Triangle",
            3: "Square",
            4: "L1",
            5: "R1",
            6: "L2",
            7: "R2",
            8: "Share",
            9: "Options",
            10: "PS",
            11: "L3",
            12: "R3",
            13: "Touchpad"
        }
        
        self.ps5_axis_map = {
            0: "Left Stick X",
            1: "Left Stick Y", 
            2: "L2 Trigger",
            3: "Right Stick X",
            4: "Right Stick Y",
            5: "R2 Trigger"
        }
        
        # Xbox controller mappings
        self.xbox_button_map = {
            0: "A",
            1: "B",
            3: "X",
            4: "Y",
            5: "RB",
            6: "L1",
            7: "R1",
            8: "L2",
            9: "R2",
            10: "Share",
            11: "Menu",
            12: "Home",
            13: "LS",
            14: "RS",
        }
        
        self.xbox_axis_map = {
            0: "Left Stick X",
            1: "Left Stick Y",
            2: "Right Stick X", 
            3: "Right Stick Y",
            4: "RT Trigger",
            5: "LT Trigger"
        }
        
        # Common D-pad mappings (same for both controllers)
        self.common_dpad_map = {
            (0, 1): "D-pad Up",
            (0, -1): "D-pad Down", 
            (-1, 0): "D-pad Left",
            (1, 0): "D-pad Right",
            (-1, 1): "D-pad Up-Left",
            (1, 1): "D-pad Up-Right",
            (-1, -1): "D-pad Down-Left",
            (1, -1): "D-pad Down-Right"
        }
    
    def _detect_controller_type(self, joystick_name):
        """Auto-detect controller type based on joystick name"""
        joystick_name = joystick_name.lower()
        
        # PS5/PS4 controller detection
        ps_keywords = ['ps5', 'ps4', 'playstation', 'dualshock', 'dualsense']
        if any(keyword in joystick_name for keyword in ps_keywords):
            return 'ps5'
        
        # Xbox controller detection
        xbox_keywords = ['xbox', 'microsoft', 'x-input', 'xinput']
        if any(keyword in joystick_name for keyword in xbox_keywords):
            return 'xbox'
        
        # Additional pattern matching
        if re.search(r'(wireless controller|dualshock|dualsense)', joystick_name):
            return 'ps5'
        elif re.search(r'(xbox|microsoft|x-input)', joystick_name):
            return 'xbox'
        
        # Default fallback
        print(f"Warning: Could not detect controller type for '{joystick_name}'. Defaulting to PS5 mapping.")
        return 'ps5'
    
    def _set_controller_mappings(self, controller_type):
        """Set button and axis mappings based on controller type"""
        if controller_type == 'xbox':
            self.button_map = self.xbox_button_map
            self.axis_map = self.xbox_axis_map
            print("Using Xbox controller mappings")
        else:  # Default to PS5
            self.button_map = self.ps5_button_map
            self.axis_map = self.ps5_axis_map
            print("Using PS5/PS4 controller mappings")
        
        self.dpad_map = self.common_dpad_map
    
    def connect(self):
        if pygame.joystick.get_count() == 0:
            print("No joystick connected!")
            return False
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        joystick_name = self.joystick.get_name()
        print(f"Connected to: {joystick_name}")
        
        # Determine controller type
        if self.controller_type:
            # Use manually specified type
            self.detected_type = self.controller_type
            print(f"Using manually specified controller type: {self.controller_type}")
        else:
            # Auto-detect controller type
            self.detected_type = self._detect_controller_type(joystick_name)
            print(f"Auto-detected controller type: {self.detected_type}")
        
        # Set appropriate mappings
        self._set_controller_mappings(self.detected_type)
        
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of hats: {self.joystick.get_numhats()}")
        print("Press Ctrl+C to exit\n")
        
        return True
    
    def start_listening(self):
        if not self.connect():
            return
            
        self.running = True
        clock = pygame.time.Clock()
        
        try:
            while self.running:
                pygame.event.pump()
                
                # Check button state changes
                for i in range(self.joystick.get_numbuttons()):
                    current_state = self.joystick.get_button(i)
                    prev_state = self.prev_buttons.get(i, False)
                    
                    if current_state and not prev_state:  # Button just pressed
                        button_name = self.button_map.get(i, f"Button {i}")
                        print(f"Button pressed: {button_name}")
                    elif not current_state and prev_state:  # Button just released
                        button_name = self.button_map.get(i, f"Button {i}")
                        print(f"Button released: {button_name}")
                    
                    self.prev_buttons[i] = current_state
                
                # Check D-pad (hat) state changes
                for i in range(self.joystick.get_numhats()):
                    hat_value = self.joystick.get_hat(i)
                    prev_hat = self.prev_hats.get(i, (0, 0))
                    
                    if hat_value != prev_hat:
                        if hat_value != (0, 0):
                            dpad_name = self.dpad_map.get(hat_value, f"D-pad {hat_value}")
                            print(f"D-pad pressed: {dpad_name}")
                        elif prev_hat != (0, 0):
                            dpad_name = self.dpad_map.get(prev_hat, f"D-pad {prev_hat}")
                            print(f"D-pad released: {dpad_name}")
                    
                    self.prev_hats[i] = hat_value
                
                # Check analog sticks and triggers with change detection
                for i in range(self.joystick.get_numaxes()):
                    axis_value = self.joystick.get_axis(i)
                    prev_value = self.prev_axes.get(i, 0.0)
                    
                    # Larger deadzone and only report significant changes
                    if abs(axis_value) > 0.2 and abs(axis_value - prev_value) > 0.1:
                        axis_name = self.axis_map.get(i, f"Axis {i}")
                        # if self.controller_type == 'xbox' and 1 == i:
                        #     axis_value = -axis_value

                        print(f"Analog input: {axis_name} = {axis_value:.2f}")
                    
                    self.prev_axes[i] = axis_value
                
                clock.tick(30)  # Reduced to 30 FPS
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
    
    def get_controller_type(self):
        """Return the detected controller type"""
        return self.detected_type
    
    def get_supported_controllers(self):
        """Return list of supported controller types"""
        return ['ps5', 'xbox']
    
    def is_connected(self):
        """Check if controller is connected and initialized"""
        return self.joystick is not None and self.joystick.get_init()


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    controller_type = None
    if len(sys.argv) > 1:
        specified_type = sys.argv[1].lower()
        if specified_type in ['ps5', 'ps4', 'xbox']:
            controller_type = 'ps5' if specified_type in ['ps5', 'ps4'] else 'xbox'
            print(f"Using specified controller type: {controller_type}")
        else:
            print(f"Unsupported controller type: {specified_type}")
            print("Supported types: ps5, ps4, xbox")
            sys.exit(1)
    
    # Create and start controller
    print("Controller Support:")
    print("- PS5/PS4 DualSense/DualShock controllers")
    print("- Xbox controllers (Xbox One, Xbox Series X/S)")
    print("- Auto-detection based on controller name")
    print("- Manual override via command line: python controller.py xbox\n")
    
    controller = Controller(controller_type=controller_type)
    controller.start_listening()