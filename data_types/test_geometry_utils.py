import numpy as np

from data_types import geometry_utils
from data_types import robot_data
from data_types import se3


def test_to_homogeneous():
  points = np.zeros((10, 3))
  assert geometry_utils.to_homogeneous(points).shape == (10, 4)


def test_adjoint_matrix():
  transform = se3.Transform(
    matrix=np.array([[1, 0, 0, -0.25],
                     [0, 0, 1, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]])
  )
  cur_vector = np.array([0, 0, 1, 0, 0, 0])
  new_vector = np.dot(geometry_utils.adjoint_matrix(transform).T, cur_vector)
  assert np.allclose(new_vector, [0, -1, 0, 0, 0, -0.25])


def test_compute_wrench():
  transform = se3.Transform(
    matrix=np.array([[1, 0, 0, -0.25],
                     [0, 0, 1, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]])
  )
  cur_vector = np.array([0, 0, 1, 0, 0, 0])
  new_vector = geometry_utils.compute_wrench(cur_vector, transform)
  assert np.allclose(new_vector, [0, -1, 0, 0, 0, -0.25])


def test_compute_twist():
  transform = se3.Transform(
    matrix=np.array([[-1, 0, 0, 4],
                     [0, 1, 0, 0.4],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])
  )
  cur_vector = np.array([2.8, 4, 0, 0, 0, -2])
  new_vector = geometry_utils.compute_twist(cur_vector, transform)
  assert np.allclose(new_vector, [-2, -4, 0, 0, 0, 2])


def test_transform_pose():
  pose = se3.Transform(xyz=[1, 2, 3])
  ref_t_cur = se3.Transform(xyz=[10, 20, 30], rot=[np.pi / 2, 0, 0])
  new_pose = geometry_utils.transform_pose(
    pose, ref_t_cur
  )
  assert np.allclose(new_pose.translation, [11, 17, 32])
  assert np.allclose(new_pose.rpy, [np.pi / 2, 0, 0])


def test_transform_twist():
  twist = np.array([1, 2, 3, 0.1, 0.2, 0.3])
  ref_t_cur = se3.Transform(xyz=[10, 20, 30], rot=[np.pi / 2, 0, 0])
  new_twist = geometry_utils.transform_twist(
    twist, ref_t_cur
  )
  assert np.allclose(new_twist, [1, -3, 2, 0.1, -0.3, 0.2])


def test_transform_wrench():
  wrench = np.array([1, 2, 3, 0.1, 0.2, 0.3])
  ref_t_cur = se3.Transform(xyz=[10, 20, 30], rot=[np.pi / 2, 0, 0])
  new_wrench = geometry_utils.transform_twist(
    wrench, ref_t_cur
  )
  assert np.allclose(new_wrench, [1, -3, 2, 0.1, -0.3, 0.2])


def test_transform_stiffness():
  ref_t_cur = se3.Transform(xyz=[10, 20, 30], rot=[np.pi / 2, 0, 0])
  stiffness = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0],
                        [0, 0, 0, 0, 5, 0],
                        [0, 0, 0, 0, 0, 6]])
  new_stiffness = geometry_utils.transform_stiffness(stiffness, ref_t_cur)
  np.allclose(new_stiffness,
              np.array([[1, 0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0, 0],
                        [0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0],
                        [0, 0, 0, 0, 6, 0],
                        [0, 0, 0, 0, 0, 5]]))


def test_tranform_state():
  pose = se3.Transform(xyz=[1, 2, 3])
  twist = np.array([1, 2, 3, 0.1, 0.2, 0.3])
  wrench = np.array([1, 2, 3, 0.1, 0.2, 0.3])
  ref_t_cur = se3.Transform(xyz=[10, 20, 30], rot=[np.pi / 2, 0, 0])
  state = robot_data.RobotState(
    timestamp=0.1,
    pose=pose,
    twist=twist,
    jpos=[0.2] * 6,
    jvel=[0.1] * 6,
    ft_data=robot_data.FtData(timestamp=0.1, wrench=wrench)
  )
  new_state = geometry_utils.transform_state(state, ref_t_cur)
  assert np.allclose(new_state.ft_data.wrench, [1, -3, 2, 0.1, -0.3, 0.2])
  assert np.allclose(new_state.twist, [1, -3, 2, 0.1, -0.3, 0.2])
  assert np.allclose(new_state.pose.translation, [11, 17, 32])
  assert np.allclose(new_state.pose.rpy, [np.pi / 2, 0, 0])


if __name__ == "__main__":
    test_transform_pose()
    test_transform_twist()
    test_transform_wrench()
    test_adjoint_matrix()
    test_compute_twist()
    test_compute_wrench()
    test_transform_stiffness()
    test_tranform_state()
