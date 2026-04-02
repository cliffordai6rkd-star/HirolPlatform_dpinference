python scripts/train_mlp_bc.py \
    --data_dir dataset/1025-pick_and_place_fr3_295ep \
    --obs_space jonit_position \
    --action_space joint_position \
    --epochs 1000 \
    --batch_size 32 \
    --rgb_encoder resnet18 \
    --share_rgb \
    --freeze_backbone \
    --loss mse \
    --save_dir /home/yuxuan/Code/hirol/bc_policy/ckpts/1025_pick_n_place_jps2jps \
    --action_pred_steps 2 \
    --skip_step_nums 2 \
    --num_episodes 50

python scripts/train_mlp_bc.py \
    --data_dir dataset/1025-pick_and_place_fr3_295ep \
    --obs_space mask \
    --action_space joint_position \
    --epochs 1000 \
    --batch_size 32 \
    --rgb_encoder resnet18 \
    --share_rgb \
    --freeze_backbone \
    --loss mse \
    --save_dir /home/yuxuan/Code/hirol/bc_policy/ckpts/1025_pick_n_place_mask2jps \
    --action_pred_steps 2 \
    --skip_step_nums 2 \
    --num_episodes 50

python scripts/train_mlp_bc.py \
    --data_dir dataset/1025-pick_and_place_fr3_295ep \
    --obs_space delta_ee_pose \
    --action_space end_effector_pose_delta \
    --epochs 1000 \
    --batch_size 32 \
    --rgb_encoder resnet18 \
    --share_rgb \
    --freeze_backbone \
    --loss mse \
    --save_dir /home/yuxuan/Code/hirol/bc_policy/ckpts/1025_pick_n_place_de2de \
    --action_pred_steps 2 \
    --skip_step_nums 2 \
    --num_episodes 50

python scripts/train_mlp_bc.py \
    --data_dir dataset/1025-pick_and_place_fr3_295ep \
    --obs_space mask \
    --action_space end_effector_pose_delta \
    --epochs 1000 \
    --batch_size 32 \
    --rgb_encoder resnet18 \
    --share_rgb \
    --freeze_backbone \
    --loss mse \
    --save_dir /home/yuxuan/Code/hirol/bc_policy/ckpts/1025_pick_n_place_mask2de \
    --action_pred_steps 2 \
    --skip_step_nums 2 \
    --num_episodes 50


