from dataset.reader import RerunEpisodeReader
import os, random
import glog as log
import numpy as np
from bc_utils.utils import get_data_stata, normalize, save_data_stat
from tqdm import tqdm

import torch as th
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class HirolImageDataset(Dataset):
    def __init__(self, data_dir, obs_space, action_space, action_prediction_steps=1,
                 skip_steps=1, rotation_transform=None, num_episodes=50, img_transforms=None,
                 img_shape=(224, 224), stata_saving_path=None):
        super().__init__()
        if not os.path.exists(data_dir):
            raise ValueError(f'Could not find the data directory: {data_dir}')
        
        self.rgb_keys = []
        self.skip_steps = skip_steps
        self.data_dir = data_dir
        self.img_shape = img_shape
        self.data_reader = RerunEpisodeReader(task_dir=data_dir, action_type=action_space,
                                              observation_type=obs_space, action_ori_type="euler",
                                              action_prediction_step=action_prediction_steps,
                                              rotation_transform=rotation_transform)
       
        default_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),  # [0,1], CHW
        ])
        tf = img_transforms or default_tf
        
        possible_episodes = os.listdir(data_dir)
        if len(possible_episodes) == 0:
            raise RuntimeError(f"No episodes found under {data_dir} (expected directories named like 'episode_XXXX').")
        
        if len(possible_episodes) < num_episodes:
            self.episode_dirs = possible_episodes
            log.warn(f'Requested {num_episodes} episodes but only found {len(possible_episodes)} in {data_dir}')
        else:
            self.episode_dirs = random.sample(possible_episodes, num_episodes)
        log.info(f'Ready to load the {self.episode_dirs} in {data_dir}!!!')
        actions = None; observations = None; cam_imgs = []
        for episode in tqdm(self.episode_dirs, desc="Processing episodes"):
            if not os.path.exists(os.path.join(data_dir, episode)):
                log.warn(f'Episode directory {episode} does not exist, skipping.')
                continue
            
            episode_id = int(episode.lstrip("episode_"))
            episode_data = self.data_reader.return_episode_data(episode_id, self.skip_steps)
            for step_data in tqdm(episode_data, desc="Processing steps"):
                if any(k not in step_data for k in ["observations", "actions", "colors"]):
                    log.warn(f'The step data for {episode} missing required keys [colors, observations, actions]')
                    continue
                
                step_action = step_data["actions"]
                cur_action = np.hstack([np.asarray(a) for _, a in step_action.items()]).astype(np.float32)
                actions = cur_action[None, :] if actions is None else np.vstack((actions, cur_action[None, :]))

                step_obs = step_data["observations"]
                cur_obs = np.hstack([np.asarray(o) for _, o in step_obs.items()]).astype(np.float32)
                observations = cur_obs[None, :] if observations is None else np.vstack((observations, cur_obs[None, :]))
                
                colors = {}
                for cam_name, img in step_data["colors"].items():
                    if cam_name not in self.rgb_keys: self.rgb_keys.append(cam_name)
                    img = tf(img)
                    # log.info(f'{cam_name} img shape: {img.shape}')
                    colors[cam_name] = img 
                cam_imgs.append(colors)
        log.info(f'Loaded dataset samples from {len(self.episode_dirs)} episodes.')
        
        # Get the data mean and std and save to a file
        obs_mean, obs_std = get_data_stata(observations)
        action_mean, action_std = get_data_stata(actions)
        self.data_dict = dict(obs_mean=obs_mean, obs_std=obs_std,
                         action_mean=action_mean, action_std=action_std)
        stata_saving_path = stata_saving_path if stata_saving_path else os.path.join(data_dir, "data_stats.json")
        save_data_stat(self.data_dict, saving_path=stata_saving_path)
        
        self.observations = observations; self.actions = actions; self.cam_imgs = cam_imgs
    
    def get_rgb_keys(self):
        return self.rgb_keys
    
    def get_dims(self):
        return self.observations.shape[1], self.actions.shape[1]
    
    def __len__(self):
        return len(self.cam_imgs)
    
    def __getitem__(self, idx):
        #""" Returns colors, observations(propio states), actions """
        cam_imgs = {k: v for k, v in self.cam_imgs[idx].items()}  # Dict[str, Tensor(C,H,W)]
        actions = self.actions[idx]
        actions = normalize(actions, self.data_dict['action_mean'], self.data_dict['action_std'])
        actions = th.tensor(actions, dtype=th.float32)
        obs = self.observations[idx]
        obs = normalize(obs, self.data_dict['obs_mean'], self.data_dict['obs_std'])
        obs = th.tensor(obs, dtype=th.float32)
        return cam_imgs, obs, actions
    
def main():
    import time
    
    # 配置参数
    data_dir = "your_data_directory"  # 替换为实际的数据目录路径
    device = "cuda" if th.cuda.is_available() else "cpu"
    batch_size = 4
    num_workers = 2  # 根据你的系统调整
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 如果需要调整大小
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    # possible transform
    img_size = (224, 224)
    transforms.Compose([
        # 转换为PIL图像（因为很多变换需要PIL图像格式）
        transforms.ToPILImage(),
        
        # 颜色/亮度增强
        transforms.ColorJitter(
            brightness=0.3,    # 亮度调整范围 (±30%)
            contrast=0.3,      # 对比度调整范围 (±30%)
            saturation=0.3,    # 饱和度调整范围 (±30%)
            hue=0.1           # 色相调整范围 (±10%)
        ),
        
        # 几何变换
        transforms.RandomRotation(degrees=15),  # 随机旋转 ±15度
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，概率50%
        
        # 随机调整大小和裁剪
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.8, 1.0),  # 随机缩放范围 80%-100%
            ratio=(0.9, 1.1)   # 宽高比范围
        ),
        
        # 转换为张量
        transforms.ToTensor(),
        
        # 标准化（使用ImageNet的均值和标准差）
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        # 创建数据集实例
        log.info("创建数据集...")
        dataset = HirolImageDataset(
            device=device,
            data_dir=data_dir,
            obs_space="your_obs_space",  # 替换为实际的观测空间
            action_space="your_action_space",  # 替换为实际的动作空间
            action_prediction_steps=1,
            skip_steps=1,
            rotation_transform=None,
            num_episodes=10,  # 测试时用较少的episode
            transforms=transform,
            img_shape=(224, 224)
        )
        
        log.info(f"数据集大小: {len(dataset)}")
        
        # 创建DataLoader
        log.info("创建DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False
        )
        
        # 测试数据加载
        log.info("开始测试数据加载...")
        start_time = time.time()
        
        for batch_idx, (cam_imgs, obs_tensor, action_tensor) in enumerate(dataloader):
            batch_time = time.time() - start_time
            
            # 打印批次信息
            log.info(f"批次 {batch_idx + 1}:")
            log.info(f"  加载时间: {batch_time:.4f}秒")
            
            # 检查摄像头图像
            log.info("  摄像头图像:")
            for cam_name, img_tensor in cam_imgs.items():
                log.info(f"    {cam_name}: {img_tensor.shape}, 设备: {img_tensor.device}, "
                        f"范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
            
            # 检查观测张量
            log.info(f"  观测张量: {obs_tensor.shape}, 设备: {obs_tensor.device}, "
                    f"范围: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")
            
            # 检查动作张量
            log.info(f"  动作张量: {action_tensor.shape}, 设备: {action_tensor.device}, "
                    f"范围: [{action_tensor.min():.3f}, {action_tensor.max():.3f}]")
            
            # 检查数据类型
            log.info(f"  数据类型 - 图像: {cam_imgs[list(cam_imgs.keys())[0]].dtype}, "
                    f"观测: {obs_tensor.dtype}, 动作: {action_tensor.dtype}")
            
            # 只测试前3个批次
            if batch_idx >= 2:
                log.info("已完成3个批次的测试")
                break
                
            start_time = time.time()  # 重置计时器
            
        log.info("数据加载器测试完成!")
    except Exception as e:
        log.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
     
if __name__ == "__main__":
    main()
    