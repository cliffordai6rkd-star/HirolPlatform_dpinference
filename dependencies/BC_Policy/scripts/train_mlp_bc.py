
import os
import argparse
import wandb
from typing import Dict
from datetime import datetime
import numpy as np

import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.reader import Action_Type_Mapping_Dict, ObservationType
from dataset.bc_dataloader import HirolImageDataset
from bc_utils.utils import LossType
from models.mlp_bc import MlpBcPolicy


def move_imgs_to_device(cam_imgs: Dict[str, th.Tensor], device: th.device) -> Dict[str, th.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in cam_imgs.items()}


def main():
    parser = argparse.ArgumentParser(description="Train MLP BC policy (vision + proprio).")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--obs_space', type=str, required=True)
    parser.add_argument('--action_space', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--share_rgb', action='store_true', default=False)
    parser.add_argument('--rgb_encoder', type=str, default='resnet18',
                        choices=['resnet18','resnet34','resnet50','resnet101','resnet152'])
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--no_pretrained', action='store_true', default=False)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512,512,512])
    parser.add_argument('--proprio_feature', type=int, default=32)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse','log'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--action_pred_steps', type=int, default=2)
    parser.add_argument('--skip_step_nums', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./ckpts_bc')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=100)
    args = parser.parse_args()

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    now = datetime.now()
    print("当前日期和时间:", now)
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    wandb_run = wandb.init(project="bc-policy",
            entity=None,                 # 组织名，可选
            mode="online",               # online/offline/disabled
            name=f"{args.rgb_encoder}-share_rgb{args.share_rgb}-propio{args.proprio_feature}-{formatted_now}",       # 自定义 run 名称
            tags=["bc", "mlp", "resnet"],
            group="exp-001",             # 将多次实验归入一个组
            job_type="train",            # 标注任务类型：train/eval/data-prepare 等
            config={                     # 记录超参：可用 run.config 访问
                "epochs": 30,
                "batch_size": 64,
                "lr": 3e-4,
                "weight_decay": 1e-5,
                "share_rgb": True,
                   },
    )
    
    # Dataset
    action_type = Action_Type_Mapping_Dict[args.action_space]
    obs_type = ObservationType(args.obs_space)
    dataset = HirolImageDataset(
        data_dir=args.data_dir,
        obs_space=obs_type,
        action_space=action_type,
        num_episodes=args.num_episodes,  # adjust as needed
        stata_saving_path=os.path.join(args.save_dir, 'data_stats.json'),
        action_prediction_steps=args.action_pred_steps,
        skip_steps=args.skip_step_nums
    )
    rgb_keys = dataset.get_rgb_keys()
    proprio_dim, action_dim = dataset.get_dims()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    loss_type = LossType.MSE if args.loss == 'mse' else LossType.LOG_PROB

    # Model
    model = MlpBcPolicy(
        rgb_keys=rgb_keys,
        share_rgb=args.share_rgb,
        rgb_encoder_name=args.rgb_encoder,
        pretrained_backbone=not args.no_pretrained,
        freeze_resnet=args.freeze_backbone,
        proprio_dim=proprio_dim,
        proprio_feature_size=args.proprio_feature,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        loss_type=loss_type,
        training=True
    ).to(device)

    # Optimizer
    optim = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0

        for i, (cam_imgs, obs, actions) in enumerate(pbar, 1):
            cam_imgs = move_imgs_to_device(cam_imgs, device)
            # for k, img in cam_imgs.items():
            #     print(f'{k} shape: {img.shape}')
            obs = obs.to(device, non_blocking=True)
            # print(f'obs shape: {obs.shape}')
            actions = actions.to(device, non_blocking=True)
            # print(f'actions shape: {actions.shape}')

            optim.zero_grad(set_to_none=True)

            if loss_type == LossType.MSE:
                pred = model(cam_imgs, obs)  # (B, Da)
                loss = model.compute_mse_loss(actions, pred)
            else:
                _, mu, logvar = model(cam_imgs, obs)
                loss = model.compute_log_prob_loss(actions, mu, logvar)

            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            loss_item = loss.item()
            running += loss_item
            if i % args.log_interval == 0:
                pbar.set_postfix(loss=running / args.log_interval)
                running = 0.0
                
            # wandb log
            if global_step % 100 == 0:
                wandb_run.log({"train/loss": loss_item, "train/lr": 
                        args.lr, "epoch": epoch, "global_step": global_step},
                        step=global_step)
                for cam_name, img in cam_imgs.items():
                    selected_id = np.random.randint(0, len(img))
                    cpu_img = img[selected_id].detach().cpu()
                    wandb_run.log({f"images/{cam_name}": wandb.Image(cpu_img)}, step=global_step)
            global_step += 1

        # Save checkpoint per epoch
        if epoch != 0 and epoch % args.save_interval == 0:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optim.state_dict(),
                'args': vars(args),
                'rgb_keys': rgb_keys,
                'proprio_dim': proprio_dim,
                'action_dim': action_dim,
            }
            th.save(ckpt, os.path.join(args.save_dir, f'bc_epoch{epoch:03d}.pt'))
        
    print("Training completed.")

if __name__ == '__main__':
    # python /mnt/data/train_bc.py \
    #   --data_dir /path/to/your/task_dir \
    #   --obs_space your_obs_space \
    #   --action_space your_action_space \
    #   --epochs 50 \
    #   --batch_size 64 \
    #   --rgb_encoder resnet18 \
    #   --share_rgb \
    #   --freeze_backbone \
    #   --loss mse
    main()