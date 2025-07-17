# 请参考[ik_cfg](./config/ik_fr3_cfg.yaml)
## 参数说明
1. `damping_weight`: 求解ik的damping系数
2. `ik_type`: 求解ik的方法选择，方法支持["gaussian_newtown", "dls", "lm"], **pyroki 慎用， 关节会有跳变**
3. `tolerance`: 求解收敛停止条件
4. `max_iteration`: 求解最大迭代此书

