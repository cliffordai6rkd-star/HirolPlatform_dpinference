#!/bin/bash

# FR3机器人ACT推理执行脚本
# 
# 频率配置说明:
# - ACT推理频率: 15Hz (每66.7ms生成一个动作指令)  
# - MotionFactory控制频率: 800Hz (每1.25ms执行一次控制循环)
# - 动作块大小: 100 (约6.7秒缓冲)
# - 采样间隔: 1 (使用所有预测动作)

python learning/demo/run_act_inference.py \
    --robot fr3 \
    --ckpt_dir learning/ckpts/fr3_lego_tower \
    --config learning/demo/configs/fr3_inference_config.yaml \
    --frequency 30.0 \
    --max_steps 12000 \
    --log_level INFO