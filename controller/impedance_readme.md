# 请参考[impedance_cfg](./config/impedance_fr3_cfg.yaml)
## 参数说明
1. `stiffness`: 刚度参数(6维向量)
2. `saturation`: 输出关节力矩的上下限
3. `kp_null`: 关节构型零空间的kp参数(向量表达，维度需为主动关节自由度一样)
4. `q_des`: 零空间关节构型的目标位置
5. `dq_damping`: 关节构型零空间的damping系数
   