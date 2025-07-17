from motion.kinematics import PinocchioKinematicsModel
import abc
from hardware.base.utils import RobotJointState, Buffer, transform_pose

class a(metaclass=abc.ABCMeta):
    def __init__(self, ):
        print("a init")
        self.a = 1
        self.b_fa = 2

    @abc.abstractmethod
    def lab(self):
        print("father class")

class b(a):
    def __init__(self):
        super().__init__()
        print("b init")
        
    def lab(self):
        super().lab()
        print("lab in child")
        print(f"b_fa = {self.b_fa}")
        self.a = 100
        print(f'a = {self.a}')
        
def func(a, b, c):
    return a +b +c

def check(item: dict):
    item['a'] = [1,2,3]

if __name__ == '__main__':
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # 定义两个四元数
    quat1 = R.from_quat(np.array([0.707, 0.0, 0.0, 0.707]))  # (w, x, y, z)
    quat2 = R.from_quat(np.array([0.5, 0.5, 0.5, 0.5]))

    # 计算四元数的差
    conjugate_quat2 = quat2.inv()
    result_quat = quat1 * conjugate_quat2

    print("Result quaternion after subtraction:", np.array(result_quat.as_quat()))
    
    buff = Buffer(10, 3)
    print(f'buffer: {buff._data}')
    buff.push_data(np.array([1.0,2.0,3.3]).tolist())
    print(f'buffer: {buff._data}')
    buff.push_data(np.array([3.0,4.0,5.3]).tolist())
    print(f'buffer: {buff._data}')
    data = buff.pop_data()
    print(f'data: {data}, buff: {buff._data}')
    data = buff.pop_data()
    print(f'data: {data}, buff: {buff._data}')
    data = buff.pop_data()
    print(f'data: {data}, buff: {buff._data}')
    
    values = [2,3]
    print(f'res: {func(*values, 4)}')
    values = [1,2,3,4,5]
    b = [values[i] for i in range(5) if i%2 == 0]
    print(b)
    
    
    world2base = [0.14, 0, 0.9, 0, 0, 0,   1.  ]
    target = [1.66384051e-01, 5.35049608e-16, 6.45220318e-01, 9.76069194e-01,
 1.15367319e-01, 1.83060513e-01, 2.16369912e-02]
    print(f'quat norm: {np.linalg.norm(target[3:])}')
    print(f'after: {transform_pose(world2base, target)}')
    
    a = dict()
    a ['a'] = 1
    a['b'] = 2
    for key, value in a.items():
        if key == 'a':
            value = 5
        print(f'a: {a}')
    print(f'a: {a}')
    
    my_dict = {}
    print(f'my dict: {my_dict}')
    check(my_dict)
    print(f'my dict: {my_dict}')
    