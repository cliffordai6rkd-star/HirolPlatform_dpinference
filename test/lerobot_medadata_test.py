from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "1113_left_fr3_insert_pinboard_53ep"   # 或 mask2q 版本
root = "factory/tasks/inferences_tasks/lerobot/ckpts/1113_left_fr3_insert_pinboard_53ep"

meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
feat = meta.features["observation.state"]
stats = meta.stats["observation.state"]  # MEAN_STD 模式下包含 mean/std

print("dataset state shape:", feat["shape"])
print("dataset state names (first 10):", feat.get("names", [])[:10])  # 可能是 ['s0','s1',...]
print("stats mean (first 10):", stats["mean"][:10])
print("stats std  (first 10):", stats["std"][:10])