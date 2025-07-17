from utilis.iks import minkowski_ik, analytical_ik, qp_ik
from pathlib import Path
def IK_solve(target_pose, current_joints):
    """
    计算逆运动学
    :param panda: Panda对象
    :param target_pose: 目标位姿
    :param current_joints: 当前关节角度
    :return: 关节角度 np.array

    添加了异常检测,防止某种ik失败，提升鲁棒性
    """
    # 获取当前关节配置
    # 不需要在这里获取，因为已经传入了current_joints参数
    
    # 找到模型路径


    # 获取当前脚本所在目录
    project_root = Path(__file__).parent.resolve()

    # 构建相对路径
    # xml_path = str(project_root / "franka_fr3" / "fr3_with_hand.xml")
    # urdf_file = str(project_root / "franka_fr3" / "fr3_franka_hand.urdf")
    xml_path = str(project_root / "assets" / "franka_fr3" / "fr3_with_hand.xml")
    urdf_file = str(project_root / "assets" / "franka_fr3" / "fr3_franka_hand.urdf")

    # 尝试各种IK方法
    try:
        # 尝试Minkowski IK 
        if xml_path:
            solution = minkowski_ik(target_pose, xml_path, current_joints)
            if solution is not None:
                return solution
    except Exception as e:
        print(f"Minkowski IK failed: {e}")

    try:
        # 尝试QP IK
        solution = qp_ik(target_pose, current_joints)
        if solution is not None:
            return solution
    except Exception as e:
        print(f"QP IK failed: {e}")

    try:
        # 尝试解析解IK 
        solution = analytical_ik(target_pose, current_joints)
        if solution is not None:
            return solution
    except Exception as e:
        print(f"Analytical IK failed: {e}")
    
    
    # try:
    #     # 尝试KDL IK 
    #     if urdf_file:
    #         solution = kdl_ik(target_pose, urdf_file, current_joints)
    #         if solution is not None:
    #             return solution
    # except Exception as e:
    #     print(f"KDL IK failed: {e}")
    

    
    # 如果所有方法都失败，返回None
    print("All IK methods failed to find a solution")
    return None
