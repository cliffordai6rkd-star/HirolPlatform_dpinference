from trajectory.trajectory_base import TrajectoryBase

class JointTrajectory(TrajectoryBase):
    def __init__(self, config, buffer):
        super().__init__(config, buffer)
        
    def plan_trajectory(self, target):
        return super().plan_trajectory(target)
    
    
        