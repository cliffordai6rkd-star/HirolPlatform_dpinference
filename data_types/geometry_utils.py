import numpy as np

from data_types import robot_data
from data_types import se3


def to_homogeneous(points: np.ndarray):
  """Appends 1s to the xyz points.

  Args:
    points: The input points, size of Nx3.
  """
  homo = np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1)
  return homo


def transform_points(points: np.ndarray, ref_t_cur: se3.Transform):
  """Transform points to reference frame.

  Args:
    points: The points in the current frame, size of 3xN.
    ref_t_cur: The transform from reference to the current frame.
  """
  return np.dot(ref_t_cur.matrix[:3, :3], points) + ref_t_cur.matrix[:3, 3:]


def transform_state(obs: robot_data.RobotState, ref_t_cur: se3.Transform
                    ) -> robot_data.RobotState:
  """Expresses the RobotState in a reference frame.

  Args:
    ref_t_cur: Transform of reference frame to current frame.
  """
  new_obs = robot_data.RobotState(
    timestamp=obs.timestamp,
    pose=transform_pose(obs.pose, ref_t_cur) if obs.pose is not None else None,
    twist=transform_twist(
      obs.twist, ref_t_cur) if obs.twist is not None else None,
    jpos=obs.jpos,
    jvel=obs.jvel,
    ft_data=robot_data.FtData(
      timestamp=obs.ft_data.timestamp,
      wrench=transform_wrench(obs.ft_data.wrench, ref_t_cur)
      ) if obs.ft_data is not None else None
  )
  return new_obs


def transform_command(cmd: robot_data.RobotCommand, ref_t_cur: se3.Transform
                      ) -> robot_data.RobotCommand:
  """Expresses the RobotCommand in a reference frame.
  """
  new_cmd = robot_data.RobotCommand(
    pose=transform_pose(cmd.pose, ref_t_cur) if cmd.pose is not None else None,
    twist=transform_twist(
      cmd.twist, ref_t_cur) if cmd.twist is not None else None,
    wrench=transform_wrench(
      cmd.wrench, ref_t_cur) if cmd.wrench is not None else None
  )
  return new_cmd


def transform_pose(pose: se3.Transform, ref_t_cur: se3.Transform
                   ) -> se3.Transform:
  """Experesses the pose in a fixed reference frame."""
  return ref_t_cur * pose


def transform_twist(twist: np.ndarray, ref_t_cur: se3.Transform) -> np.ndarray:
  """Experesses a body twist in a fixed reference frame.

  The twist is rotated such that it is aligned with the reference frame.
  """
  linear = np.dot(ref_t_cur.matrix[:3, :3], twist[:3])
  angular = np.dot(ref_t_cur.matrix[:3, :3], twist[3:])
  return np.concatenate((linear, angular))


def compute_twist(twist: np.ndarray, desired_t_cur: se3.Transform
                  ) -> np.ndarray:
  """Computes the twist of the desired body frame given the current twist.

  The two body frames are assumed to be rigidly attached with the transform
  given by desired_t_cur.
  """
  return np.dot(adjoint_matrix(desired_t_cur), twist)


def compute_twist_in_fixed(twist: np.ndarray,
                           desired_t_cur: se3.Transform,
                           fixed_t_desired: se3.Transform):
  """Computes the twist of a desired body expressed in a fixed frame.

  Given the twist of the current body expressed in a fixed frame,
  compute the twist of a desired body expressed in the same fixed frame.
  Typically used for converting a twist from one body to another rigidly
  attached body, both expressed in base frame.
  """
  rotation = np.eye(6)
  rotation[:3, 3:] = np.dot(
    np.dot(fixed_t_desired.matrix[:3, :3], skew_symmetric_matrix(
      desired_t_cur.translation)), fixed_t_desired.inverse().matrix[:3, :3])
  return np.dot(rotation, twist)


def transform_wrench(wrench: np.ndarray, ref_t_cur: se3.Transform
                     ) -> np.ndarray:
  """Expresses a body wrench in a fixed reference frame.

  The wrench is rotated such that it is aligned with the reference frame.
  """
  linear = np.dot(ref_t_cur.matrix[:3, :3], wrench[:3])
  angular = np.dot(ref_t_cur.matrix[:3, :3], wrench[3:])
  return np.concatenate((linear, angular))


def compute_wrench(wrench: np.ndarray, desired_t_cur: se3.Transform
                   ) -> np.ndarray:
  """Computes the wrench of the desired body frame given the current wrench.

  The two body frames are assumed to be rigidly attached with the transform
  given by desired_t_cur.
  """
  return np.dot(adjoint_matrix(desired_t_cur).T, wrench)


def transform_stiffness(stiffness: np.ndarray, ref_t_cur: se3.Transform
                        ) -> np.ndarray:
  rotation = np.zeros((6, 6))
  rotation[:3, :3] = ref_t_cur.matrix[:3, :3]
  rotation[3:, 3:] = ref_t_cur.matrix[:3, :3]

  return np.dot(np.dot(rotation, stiffness), rotation.T)


def skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
  return np.array([[0, -vector[2], vector[1]],
                   [vector[2], 0, -vector[0]],
                   [-vector[1], vector[0], 0]])


def adjoint_matrix(transform: se3.Transform) -> np.ndarray:
  matrix = np.zeros((6, 6))
  matrix[:3, :3] = transform.matrix[:3, :3]
  matrix[3:, 3:] = transform.matrix[:3, :3]
  matrix[:3, 3:] = np.dot(
    skew_symmetric_matrix(transform.translation), transform.matrix[:3, :3])

  return matrix
