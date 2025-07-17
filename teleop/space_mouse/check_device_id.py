import pyspacemouse
import time

NUM_ID_PER_DEVICE = 3
DEVICE_NAME = None

def check_device_id(device_id):
    """
        Check if the device ID belongs to which physical device.
        :param device_id: The ID of the device to check.
    """
    print(f'Tring to open device ID {device_id}')
    mouse = pyspacemouse.open(device=DEVICE_NAME, DeviceNumber=device_id)
    if mouse is None:
        print(f"Failed to open device with ID {device_id}.")
        return
    print(f"Device ID {device_id} opened successfully and ready to be checked.")
    time.sleep(3)
    
    start_time = time.time()
    while True:
        state = mouse.read()
        if state:
            print(f"Device ID {device_id} state: x={state.x}, y={state.y}, z={state.z}, "
                    f"rx={state.roll}, ry={state.pitch}, rz={state.yaw}, "
                    f"button 1 state: {state.buttons[0]}, button 2 state: {state.buttons[1]}")
            print(f"")
            
        if time.time() - start_time > 5:
            mouse.close()
            print(f"Exiting check for device ID {device_id}.")
            break
            
    
    # while True:
    #     user_input = input(f"Press 'c' to continue checking device ID {device_id}, or 'q' to quit: ")
    #     if user_input.lower() == 'c':
    #         state = mouse.read()
    #         if state:
    #             print(f"Device ID {device_id} state: x={state.x}, y={state.y}, z={state.z}, "
    #                   f"rx={state.roll}, ry={state.pitch}, rz={state.yaw}")
    #             print(f"Button 1 state: {state.buttons[0]}, button 2 state: {state.buttons[1]}")
    #         else:
    #             print(f"No data received from device ID {device_id}.")
    #         print('Please touch your mouse to check!')
    #     elif user_input.lower() == 'q':
    #         print(f"Exiting check for device ID {device_id}.")
    #         break
    #     else:
    #         print("Invalid input, please press 'c' to continue or 'q' to quit.")

def main():
    devices = pyspacemouse.list_devices()
    print(f'Your PC has connected {len(devices)/NUM_ID_PER_DEVICE} 3D muse devices.')
    
    if not devices:
        print("No 3D mice found.")
        return
    
    global DEVICE_NAME
    DEVICE_NAME = devices[0]  # Use the first device as default
    
    for i in range(len(devices)):
        check_device_id(i)
        print(f"Device ID {i} is checked!!!")
    
    print("All devices checked. Exiting...")

if __name__ == "__main__":
    main()