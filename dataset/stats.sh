#!/bin/bash
# 统计并按文件数排序

BASE_DIR="$1"

echo "Episode Colors File Count (sorted by file count)"
echo "================================================="
echo ""

# 收集数据
data=$(for d in $BASE_DIR/episode_*/; do 
    count=$(ls $d/colors 2>/dev/null | wc -l)
    echo "$(basename $d) $count"
done)

# 显示排序后的结果
echo "$data" | sort -k2 -nr | awk '{printf "%-20s %10s files\n", $1, $2}'

# 统计信息
echo ""
echo "================================================="
total=$(echo "$data" | awk '{sum+=$2} END {print sum}')
count=$(echo "$data" | wc -l)
avg=$((total / count))

echo "Total Files: $total"
echo "Average: $avg files per episode"