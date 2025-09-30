# 针对你的ACT项目数据结构
import rerun as rr
import h5py
import numpy as np
import argparse
import os

def visualize_act_data(hdf5_path):
    """可视化ACT格式的HDF5数据，特别关注夹爪数据"""
    rr.init("act_gripper_analyzer", spawn=True)

    print(f"📂 Loading {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        print("📊 HDF5 structure:")
        def print_structure(name, obj):
            print(f"   {name}: {obj.shape if hasattr(obj, 'shape') else type(obj)}")
        f.visititems(print_structure)

        # 检查数据结构
        if '/observations/qpos' in f:
            qpos = f['/observations/qpos'][...]
            print(f"\n🤖 qpos shape: {qpos.shape}")
            print(f"   📊 Joint positions (first 5 steps):")
            for i in range(min(5, len(qpos))):
                joints_str = ', '.join([f"{x:.3f}" for x in qpos[i][:7]])
                print(f"      Step {i}: joints=[{joints_str}], gripper={qpos[i][7]:.3f}")

            # 专门分析夹爪数据
            gripper_positions = qpos[:, 7]  # 第8维是夹爪
            print(f"\n🦾 Gripper analysis:")
            print(f"   Min: {gripper_positions.min():.4f}")
            print(f"   Max: {gripper_positions.max():.4f}")
            print(f"   Mean: {gripper_positions.mean():.4f}")
            print(f"   Std: {gripper_positions.std():.4f}")
            print(f"   Unique values: {len(np.unique(gripper_positions))}")

            # 检查是否所有值都相同
            if gripper_positions.std() < 1e-6:
                print(f"   ⚠️  WARNING: Gripper values are constant!")

            # 可视化时间序列
            for t in range(len(qpos)):
                rr.set_time("step", sequence=t)

                # 记录所有关节位置
                for i in range(7):
                    rr.log(f"joints/joint_{i}", rr.Scalars(qpos[t, i]))

                # 特别突出夹爪位置
                rr.log("gripper/position", rr.Scalars(gripper_positions[t]))
                rr.log("gripper/position_raw", rr.Scalars(gripper_positions[t] * 0.08))  # 反归一化

        # 检查动作数据
        if '/action' in f:
            actions = f['/action'][...]
            print(f"\n🎯 Actions shape: {actions.shape}")

            if actions.shape[1] >= 8:
                action_gripper = actions[:, 7]
                print(f"\n🦾 Action gripper analysis:")
                print(f"   Min: {action_gripper.min():.4f}")
                print(f"   Max: {action_gripper.max():.4f}")
                print(f"   Mean: {action_gripper.mean():.4f}")
                print(f"   Std: {action_gripper.std():.4f}")

                if action_gripper.std() < 1e-6:
                    print(f"   ⚠️  WARNING: Action gripper values are constant!")

                # 可视化动作
                for t in range(len(actions)):
                    rr.set_time("step", sequence=t)
                    rr.log("actions/gripper", rr.Scalars(action_gripper[t]))
                    rr.log("actions/gripper_raw", rr.Scalars(action_gripper[t] * 0.08))

        # 检查并显示图像数据
        image_groups = ['/observations/images']
        for group_path in image_groups:
            if group_path in f:
                image_group = f[group_path]
                print(f"\n📷 Found images in {group_path}:")

                # 获取所有相机
                cameras = list(image_group.keys())
                print(f"   Cameras: {cameras}")

                for cam_name in cameras:
                    if cam_name in image_group:
                        # 获取图像数据集但不加载到内存
                        image_dataset = image_group[cam_name]
                        num_frames = len(image_dataset)
                        print(f"   📸 {cam_name}: {image_dataset.shape}")

                        # 可视化图像序列 - 使用索引访问避免内存问题
                        print(f"      Processing {num_frames} frames...")
                        for t in range(num_frames):
                            rr.set_time("step", sequence=t)
                            # 直接从HDF5索引访问单帧，避免一次性加载所有数据
                            rr.log(f"cameras/{cam_name}", rr.Image(image_dataset[t]))

def main():
    parser = argparse.ArgumentParser(description='Visualize ACT HDF5 data with gripper focus')
    parser.add_argument('--hdf5_path', type=str, default='test/viz/fr3_bs_seg_overlap/episode_38.hdf5',
                        help='Path to HDF5 file')
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_path):
        print(f"❌ File not found: {args.hdf5_path}")
        return

    visualize_act_data(args.hdf5_path)

if __name__ == "__main__":
    main()