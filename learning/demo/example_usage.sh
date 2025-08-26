#!/bin/bash

# ACT真机推理示例使用脚本
# 
# 使用前请确保：
# 1. 已有训练好的ACT模型检查点
# 2. FR3机器人硬件连接正常  
# 3. 相机传感器工作正常

echo "🚀 ACT真机推理演示"
echo "=================="

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <模型检查点目录路径>"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/your/act_model"
    echo "  $0 ./models/fr3_act_peg_in_hole"
    echo ""
    exit 1
fi

CKPT_DIR=$1
CONFIG_FILE="learning/demo/configs/fr3_inference_config.yaml"

# 检查文件存在性
if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ 错误: 检查点目录不存在: $CKPT_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查必要的检查点文件
if [ ! -f "$CKPT_DIR/policy_best.ckpt" ]; then
    echo "❌ 错误: 找不到策略文件: $CKPT_DIR/policy_best.ckpt"
    exit 1
fi

if [ ! -f "$CKPT_DIR/dataset_stats.pkl" ]; then
    echo "❌ 错误: 找不到数据集统计文件: $CKPT_DIR/dataset_stats.pkl"
    exit 1
fi

echo "✅ 检查点验证通过"
echo "   - 检查点目录: $CKPT_DIR"
echo "   - 配置文件: $CONFIG_FILE"
echo ""

# 安全提醒
echo "⚠️  安全提醒："
echo "   - 确保机器人工作空间内无人员和障碍物"
echo "   - 准备好紧急停止按钮"
echo "   - 可随时按Ctrl+C安全停止"
echo ""

read -p "按回车键继续，或Ctrl+C取消..."

echo ""
echo "🏃 启动ACT推理..."

# 运行推理脚本
python learning/demo/run_act_inference.py \
    --robot fr3 \
    --ckpt_dir "$CKPT_DIR" \
    --config "$CONFIG_FILE" \
    --frequency 10.0 \
    --max_steps 1000 \
    --log_level INFO

echo ""
echo "✅ 推理完成"