#!/bin/bash

# Monte01双臂机器人ACT推理启动脚本
# 使用duo_xarm_duo_ik配置，支持16 DOF双臂控制

echo "🤖 Monte01双臂ACT推理演示"
echo "================================"

CKPT_DIR=learning/ckpts/monte01_peg_in_hole
CONFIG_FILE="learning/demo/configs/monte01_dual_arm_inference_config.yaml"

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

# Monte01双臂特有的安全提醒
echo "⚠️  Monte01双臂操作安全提醒："
echo "   - 确保双臂工作空间内无人员和障碍物"
echo "   - 左右臂工作空间可能重叠，注意避让"
echo "   - 准备好紧急停止按钮"
echo "   - 三个相机视野应畅通无阻"
echo "   - 可随时按Ctrl+C安全停止"
echo ""

echo "🔧 Monte01双臂配置信息："
echo "   - 自由度: 16 DOF (左臂8 + 右臂8)"
echo "   - 相机数量: 3个 (left_ee_cam, right_ee_cam, third_person_cam)"
echo "   - 夹爪/吸盘范围: 0-0.074m"
echo "   - 控制频率: 5Hz (适应硬件性能)"
echo "   - 简化模式: 无碰撞检测（依赖ACT模型学习的安全行为）"
echo ""

read -p "确认以上信息无误，按回车键继续，或Ctrl+C取消..."

echo ""
echo "🏃 启动Monte01双臂ACT推理..."

# 使用现有的run_act_inference.py，指定monte01机器人类型
python learning/demo/run_act_inference.py \
    --robot monte01 \
    --ckpt_dir "$CKPT_DIR" \
    --config "$CONFIG_FILE" \
    --frequency 5.0 \
    --max_steps 1500 \
    --log_level INFO

RESULT=$?

echo ""
if [ $RESULT -eq 0 ]; then
    echo "✅ Monte01双臂推理完成"
else
    echo "❌ Monte01双臂推理异常结束 (exit code: $RESULT)"
    echo ""
    echo "常见问题排查:"
    echo "  1. 检查模型是否为16 DOF双臂训练"
    echo "  2. 确认Monte01硬件连接正常"
    echo "  3. 验证三个相机工作状态"
    echo "  4. 查看日志了解具体错误信息"
fi