from hardware.base.tool_base import ToolBase
from panda_py.libfranka import Gripper
from hardware.base.utils import ToolState, ToolType, ToolControlMode
import os, yaml, time, threading
import numpy as np
import copy
import glog as log
class FrankaHand(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER
    def __init__(self, config):
        self._ip = config["ip"]
        self._grasp_force = config["grasp_force"]
        self._grasp_speed = config["grasp_speed"]
        self._epsilon_inner = config.get("epsilon_inner", 0.02)
        self._epsilon_outer = config.get("epsilon_outer", 0.06)
        super().__init__(config)
        self._state._tool_type = self._tool_type
        self._state._position = 0.08
        self._state._is_grasped = False
        self._gripper_idle = True
        self._last_command = 1.0
        self._thread_running = True
        self._update_thread = threading.Thread(target=self.update_state)
        self._lock = threading.Lock()

        # 自动检测teleop模式，避免手动注释
        is_teleop = self._is_teleop_mode()
        log.info(f"FrankaHand ({self._ip}) 模式检测: {'teleop' if is_teleop else 'normal'}")

        if not is_teleop:
            self._update_thread.start()
            log.info(f"夹爪状态更新线程已启动 ({self._ip})")
        else:
            log.info(f"检测到teleop模式，跳过夹爪状态更新线程启动 ({self._ip})")

    def _is_teleop_mode(self) -> bool:
        """
        检测是否在teleop模式下运行

        Returns:
            bool: True表示teleop模式，False表示其他模式
        """
        import sys
        import inspect

        # 方法1: 检查调用栈中是否有teleop相关模块
        for frame_info in inspect.stack():
            filename = frame_info.filename
            if any(keyword in filename.lower() for keyword in ['teleop', 'teleoperation']):
                log.debug(f"调用栈检测到teleop模式: {filename}")
                return True

        # 方法2: 检查已导入的模块中是否有teleop相关
        for module_name in sys.modules.keys():
            if any(keyword in module_name.lower() for keyword in ['teleop', 'teleoperation']):
                log.debug(f"模块检测到teleop模式: {module_name}")
                return True

        # 方法3: 检查命令行参数（如果可用）
        try:
            import sys
            if hasattr(sys, 'argv'):
                for arg in sys.argv:
                    if any(keyword in arg.lower() for keyword in ['teleop', 'teleoperation']):
                        log.debug(f"命令行参数检测到teleop模式: {arg}")
                        return True
        except:
            pass

        # 方法4: 检查环境变量
        teleop_env = os.environ.get('TELEOP_MODE', '').lower()
        if teleop_env in ['true', '1', 'yes']:
            log.debug("环境变量检测到teleop模式: TELEOP_MODE=true")
            return True

        return False

    def initialize(self):
        if self._is_initialized:
            return True
        
        self._gripper = Gripper(self._ip)
        # move to home
        self._gripper.homing()
        state = self._gripper.read_once()
        self._max_width = state.max_width
        self._state._position = self._max_width
        log.info(f'max width for franka hand {self._ip}: {self._max_width}')
        return True
    
    def set_hardware_command(self, command):
        if not self._is_initialized:
            raise ValueError(f'Franka hand {self._ip} is not correctly initialized!!')
        
        if np.isclose(self._last_command, command):
            return True
        
        if not self._gripper_idle:
            # log.warn(f'Franka hand {self._ip} is not idle for new command {command}')
            return False
        
        command = np.clip(command, 0 ,1)
        target = self._max_width * command
        log.info(f'🔧 FrankaHand: command={command:.3f}, target_width={target:.4f}m, max_width={self._max_width:.4f}m')
        def grasp_task():
            self._gripper_idle = False
            
            # update state
            self._lock.acquire()            
            self._state._position = target
            self._state._time_stamp = time.perf_counter()
            self._lock.release()
            
            if np.isclose(target, self._max_width):
                log.info(f'🔓 Executing gripper.move to OPEN: width={self._max_width:.4f}m')
                self._gripper.move(self._max_width, self._grasp_speed)
            else:
                if self._control_mode == ToolControlMode.INCREMENTAL:
                    log.info(f'📍 Executing gripper.move (INCREMENTAL): target={target:.4f}m')
                    self._gripper.move(target, self._grasp_speed)
                else:
                    log.info(f'✊ Executing gripper.grasp: target={target:.4f}m, force={self._grasp_force}N')
                    self._gripper.grasp(target, self._grasp_speed, self._grasp_force,
                                        self._epsilon_inner, self._epsilon_outer)
            # self._gripper.move(target, self._grasp_speed)
            self._last_command = command
            self._gripper_idle = True
            
        gripper_thread = threading.Thread(target=grasp_task)
        gripper_thread.start()
        if self._control_mode == ToolControlMode.INCREMENTAL:
            gripper_thread.join()
        return True
    
    def recover(self):
        return self._gripper.homing()
    
    def get_tool_state(self):
        self._lock.acquire()
        state = copy.deepcopy(self._state)
        self._lock.release()
        return state
    
    def update_state(self):
        log.info(f'Starting state updating thread for franka hand {self._ip}')
        
        last_read_time = time.time()
        read_frequency = 1
        while self._thread_running:
            state = self._gripper.read_once()
            self._lock.acquire()
            self._state._position = state.width
            self._state._is_grasped = state.is_grasped
            self._state._time_stamp = time.perf_counter()
            self._lock.release()
            
            dt = time.time() - last_read_time
            if dt < 1.0 / read_frequency:
                sleep_time = 1.0 / read_frequency - dt
                time.sleep(sleep_time)
            elif dt > 1.3 / read_frequency:
                log.warn(f'The franka hand {self._ip} could not reach the update thread frequency!!')
            last_read_time = time.time()
        log.info(f'franka hand {self._ip} stopped its update thread!!!!')
            
    def stop_tool(self):
        self._thread_running = False
        # self._update_thread.join()
        self._gripper.stop()
        log.info(f'Franka hand {self._ip} successfully stopped!!!!')
        
    def _set_binary_command(self, target: float) -> bool:
        """
        二进制夹爪命令接口，兼容智能夹爪控制器
        
        Args:
            target: 目标值 (0.0=关闭/抓取, 1.0=打开)
            
        Returns:
            bool: 命令是否成功执行
        """
        try:
            return self.set_hardware_command(target)
        except Exception as e:
            log.error(f"FrankaHand二进制命令执行失败: {e}")
            return False
    
    def get_tool_type_dict(self):
        tool_type_dict = {'single': self._tool_type}
        return tool_type_dict
    
if __name__ == '__main__':
    fr3_cfg = "hardware/fr3/config/fr3_cfg.yaml"
    config = "hardware/fr3/config/franka_hand_cfg.yaml"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, "../..", config)
    fr3_cfg = os.path.join(cur_path, "../..", fr3_cfg)
    print(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'yaml data: {config}')
    with open(fr3_cfg, 'r') as stream:
        fr3_cfg = yaml.safe_load(stream)["fr3"]
    
    fr3_hand = FrankaHand(config=config["franka_hand"])
    from hardware.fr3.fr3_arm import Fr3Arm
    fr3 = Fr3Arm(fr3_cfg)
    fr3._fr3_robot.teaching_mode(True)
    
    while True:
        input_data = input('Please enter c for close, o to open: ')
        if input_data == 'c':
            fr3_hand.close()
        elif input_data == 'o':
            # fr3_hand.set_hardware_command(1.0)
            fr3_hand.set_tool_command(1.0)
        gripper_state = fr3_hand.get_tool_state()
        print(f'gri: {gripper_state._position}, is grasped: {gripper_state._is_grasped}')
        time.sleep(0.01)
    