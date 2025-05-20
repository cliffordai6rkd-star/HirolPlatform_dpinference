"""
Kinematics model for a robot.

Copyright 2020 Wenzhao Lian. All rights reserved.
"""
from typing import Text, Union, Mapping, Sequence

import urdf_parser_py.urdf as urdf_parser
# from trac_ik_python.trac_ik import IK
import PyKDL as kdl
import numpy as np
import glog as log

from data_types import se3

class KinematicsModel(object):

  def __init__(self, urdf: Text,
               base_link: Text,
               ee_link: Text,
               joint_names: Sequence[Text]):
    self.urdf = urdf
    self.urdf_robot = urdf_parser.URDF.from_xml_string(urdf)

    min_joints = []
    max_joints = []
    for joint_name in joint_names:
      for joint in self.urdf_robot.joints:
        if joint.name == joint_name:
          if joint.limit is None:
            raise AttributeError('{} has no limits specified!'.format(joint.name))
          min_joints.append(joint.limit.lower)
          max_joints.append(joint.limit.upper)
          break
    if len(joint_names) != len(min_joints):
      raise AttributeError('The urdf does not contain full information ' +
                           'of all joints!')
    self.min_joints = np.array(min_joints)
    self.max_joints = np.array(max_joints)

    status, tree = treeFromString(self.urdf)
    # self.ik_pos_solver = IK("base_link",
    #                         "hand_link",
    #                         urdf_string=self.urdf)
    self.chain = tree.getChain(base_link, ee_link)

    self.num_dof = self.chain.getNrOfJoints()

    self.fk_pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
    self.fk_vel_solver = kdl.ChainFkSolverVel_recursive(self.chain)
    self.ik_vel_solver = kdl.ChainIkSolverVel_pinv(self.chain)
    self.ik_pos_solver = kdl.ChainIkSolverPos_NR(
      self.chain, self.fk_pos_solver, self.ik_vel_solver)

    self._q = kdl.JntArray(self.num_dof)
    self._q_vel = kdl.JntArrayVel(self.num_dof)
    self._default_seed = [0.0] * self.num_dof
    self.seeds = get_seeds(self.num_dof)

  # def ik(self, pose, seed=None):
  #   """Computes joint position given end-effector pose via trac_ik."""
  #   seed = self._default_seed if seed is None else seed
  #   xyz_quat = pose.to_list()
  #   jnt = self.ik_pos_solver.get_ik(seed, *xyz_quat)
  #   return jnt

  def ik(self, pose: se3.Transform, seed: np.ndarray = None) -> np.ndarray:
    """Computes joint position given end-effector pose via KDL.

    Reference:
    https://github.com/AcutronicRobotics/gym-gazebo2/blob/545e203e07895927fd7aae66596d83d97ee89fe5/gym_gazebo2/utils/general_utils.py/
    """
    if seed is None:
      # use the midpoint of the joint limits as the guess
      lowerLim = np.where(np.isfinite(self.min_joints), self.min_joints, 0.)
      upperLim = np.where(np.isfinite(self.max_joints), self.max_joints, 0.)
      seed = (lowerLim + upperLim) / 2.0
      seed = np.where(np.isnan(seed), [0.] * len(seed), seed)
      log.info("Using default seed: %s" % str(seed))

    q_kdl = kdl.JntArray(self.num_dof)
    seed_kdl = jointListToKdl(seed)
    if self.ik_pos_solver.CartToJnt(
      seed_kdl, transform_to_kdl(pose), q_kdl) >= 0:
      log.debug("IK solution found!")
      return jointKdlToList(q_kdl)
    else:
      log.warn("IK solution NOT  found!")
      return None

  def ik_vel(self, twist: Union[kdl.Twist, np.ndarray],
             jnt: np.ndarray) -> np.ndarray:
    """Computes joint velocity given end-effector twist.

    Parameters
    ----------
    twist : kdl.Twist or np.ndarray
    jnt : np.array

    Returns
    -------
    jv : np.array(float)
    """
    if not isinstance(twist, kdl.Twist):
      twist_kdl = kdl.Twist(kdl.Vector(twist[0], twist[1], twist[2]),
                            kdl.Vector(twist[3], twist[4], twist[5]))
    else:
      twist_kdl = twist

    self._set_joint(jnt)
    jv = kdl.JntArray(self.num_dof)
    if self.ik_vel_solver.CartToJnt(self._q, twist_kdl, jv) != 0:
      log.warn("The ik_vel solution might be wrong!")

    return np.array([jv[i] for i in range(jv.rows())])

  def _set_joint(self, jnt):
    """Converts numpy array to JntArray
    """
    for i in range(self.num_dof):
      self._q[i] = jnt[i]

  def _set_joint_vel(self, jp, jv):
    """Converts numpy arrays to JntArray.
    """
    for i in range(self.num_dof):
      self._q_vel.qdot[i] = jv[i]
      self._q_vel.q[i] = jp[i]

  def fk(self, jnt: np.ndarray, in_kdl: bool = False
         ) -> Union[kdl.Frame, se3.Transform]:
    """Computes end-effector pose given joint positions.

    Parameters
    ----------
    jnt : np.array(float)

    Returns
    -------
    frame : kdl.Frame or se3.Transform
    """
    self._set_joint(jnt)
    frame = kdl.Frame()
    self.fk_pos_solver.JntToCart(self._q, frame)

    if in_kdl:
      return frame
    else:
      return kdl_to_transform(frame)

  def fk_all(self, jnt: np.ndarray) -> Mapping[Text, se3.Transform]:
    """Computes poses of all links given joint position.

    NOTE: Use with caution. The correspondence between KDL.segment
    and joints is a bit confusing, particular when there are fixed joints.

    Parameters
    ----------
    jnt : np.array(float)

    Returns
    -------
    name_to_pose : {str : se3.Transform}
    """
    self._set_joint(jnt)
    frame = kdl.Frame()
    name_to_pose = {}
    for i in range(self.num_dof):
      f_name = str(self.chain.getSegment(i).getName())
      self.fk_pos_solver.JntToCart(self._q, frame, i + 1)
      name_to_pose[f_name] = kdl_to_transform(frame)

    return name_to_pose

  def fk_vel(self, jp: np.ndarray, jv: np.ndarray,
             in_kdl: bool = False) -> Union[kdl.Twist, np.ndarray]:
    """Computes end-effector twist given joint position and velocity.

    Returns
    -------
    kdl.Twist or np.ndarray
    """
    self._set_joint_vel(jp, jv)
    f = kdl.FrameVel()
    if self.fk_vel_solver.JntToCart(self._q_vel, f) != 0:
      log.warn("The fk_vel solution might be wrong!")
    twist = f.GetTwist()
    if in_kdl:
      return twist
    else:
      return np.array([twist[0], twist[1], twist[2],
                       twist[3], twist[4], twist[5]])


def kdl_to_transform(kdl_frame: kdl.Frame) -> se3.Transform:
  """Converts kdl frame to inhouse Transform.
  """
  quat = kdl_frame.M.GetQuaternion()
  trans = (kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z())

  return se3.Transform(xyz=trans, rot=quat[3:] + quat[:3])


def treeFromUrdfModel(robot_model: Text, quiet: bool = False) -> kdl.Tree:
  """Construct a PyKDL.Tree from an URDF model from urdf_parser_python.

  Args:
    robot_model: URDF xml string, ``str``
    quiet: If true suppress messages to stdout, ``bool``
  """

  root = robot_model.link_map[robot_model.get_root()]

  if root.inertial and not quiet:
    print("The root link %s has an inertia specified in " +
          "the URDF, but KDL does not support a root link " +
          "with an inertia.  As a workaround, you can add " +
          "an extra dummy link to your URDF." % root.name)

  ok = True
  tree = kdl.Tree(root.name)

  #  add all children
  for (joint, child) in robot_model.child_map[root.name]:
    if not _add_children_to_tree(
            robot_model, robot_model.link_map[child], tree):
      ok = False
      break

  return (ok, tree)


def treeFromString(xml: Text) -> kdl.Tree:
  """Construct a PyKDL.Tree from an URDF xml string.

  Args:
    xml: URDF xml string, ``str``
  """

  return treeFromUrdfModel(urdf_parser.URDF.from_xml_string(xml))


def _add_children_to_tree(robot_model, root, tree):

  # constructs the optional inertia
  inert = kdl.RigidBodyInertia(0)
  if root.inertial:
    inert = _toKdlInertia(root.inertial)

  # constructs the kdl joint
  (parent_joint_name, parent_link_name) = robot_model.parent_map[root.name]
  parent_joint = robot_model.joint_map[parent_joint_name]

  # construct the kdl segment
  sgm = kdl.Segment(
      root.name,
      _toKdlJoint(parent_joint),
      _toKdlPose(parent_joint.origin),
      inert)

  # add segment to tree
  if not tree.addSegment(sgm, parent_link_name):
    return False

  if root.name not in robot_model.child_map:
    return True

  children = [robot_model.link_map[l]
              for (j, l) in robot_model.child_map[root.name]]

  # recurslively add all children
  for child in children:
    if not _add_children_to_tree(robot_model, child, tree):
      return False

  return True


def _toKdlJoint(jnt):
  """Reference:
  https://github.com/RethinkRobotics/baxter_pykdl/blob/master/src/baxter_kdl/kdl_parser.py
  """

  # fixed = lambda j, F: kdl.Joint(j.name, kdl.Joint.None)
  def fixed(j, F):
    return kdl.Joint(j.name)  # defaul to kdl.Joint.None

  # rotational = lambda j, F: kdl.Joint(
  # j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.RotAxis)
  def rotational(j, F):
    return kdl.Joint(
        j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.RotAxis)

  # translational = lambda j, F: kdl.Joint(
  # j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.TransAxis)

  def translational(j, F):
    return kdl.Joint(
        j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.TransAxis)

  type_map = {
      'fixed': fixed,
      'revolute': rotational,
      'continuous': rotational,
      'prismatic': translational,
      'floating': fixed,
      'planar': fixed,
      'unknown': fixed,
  }

  return type_map[jnt.type](jnt, _toKdlPose(jnt.origin))


def _toKdlPose(pose):
  # URDF might have RPY OR XYZ unspecified. Both default to zeros
  rpy = pose.rpy if pose and pose.rpy and len(pose.rpy) == 3 else [0, 0, 0]
  xyz = pose.xyz if pose and pose.xyz and len(pose.xyz) == 3 else [0, 0, 0]

  return kdl.Frame(
      kdl.Rotation.RPY(*rpy),
      kdl.Vector(*xyz))


def _toKdlInertia(i):
  # kdl specifies the inertia in the reference frame of the link, the urdf
  # specifies the inertia in the inertia reference frame
  origin = _toKdlPose(i.origin)
  inertia = i.inertia
  return origin.M * kdl.RigidBodyInertia(
      i.mass, origin.p,
      kdl.RotationalInertia(
          inertia.ixx, inertia.iyy, inertia.izz, inertia.ixy,
          inertia.ixz, inertia.iyz))


def jointListToKdl(q00: np.ndarray) -> kdl.JntArray:
    """ Return KDL JntArray converted from list q00 """
    if q00 is None:
      return None
    if isinstance(q00, np.matrix) and q00.shape[1] == 0:
      q00 = q00.T.tolist()[0]
    qKdl = kdl.JntArray(len(q00))
    for i, qi0 in enumerate(q00):
      qKdl[i] = qi0
    return qKdl


def jointKdlToList(q00: kdl.Joint) -> np.ndarray:
    """ Return list converted from KDL JntArray"""
    if q00 is None:
      return None
    return np.array([q00[i] for i in range(int(q00.rows()))])


def transform_to_kdl(pose: se3.Transform) -> kdl.Frame:
  xyz_quat = pose.to_list()
  pos = kdl.Vector(xyz_quat[0], xyz_quat[1], xyz_quat[2])
  rot = kdl.Rotation()
  rot = rot.Quaternion(xyz_quat[4], xyz_quat[5], xyz_quat[6],
                       xyz_quat[3])
  return kdl.Frame(rot, pos)


def get_seeds(num_dof):
  if num_dof == 6:
    return np.array([
      [0, 0, 0, 0, 0, 0],
      [2.054957604873219, 0.5252008713772222, -1.4698621957715772,
       0.4360208223251284, 1.96793687518143, 5.0992699194],
      [2.701243330680604, 0.6888280370852469, -1.5655154807167144,
       0.8767527081155055, 1.571934159034335, 5.688003571],
      [1.4915789601495302, 0.5293272578584263, -1.1010742979091628,
       0.2488252304415847, 1.8703669998810948, 4.0508069],
      [0.8495529805819061, 0.19993705522046118, -0.8024446339723311,
       0.6622788778703272, 1.691766428486858, 4.2259843353],
      [1.5800923896068626, 0.10203470717597328, -0.6669046970282044,
       0.9292354245883816, 1.6072832294659474, 4.30933847830],
      [0.9496985991448699, -0.5406503069467344, -0.2009704561671725,
       0.5666888254433984, 0.42006316769378793, 3.550002194817929],
      [1.3777838996999328, 0.09164474790820534, -1.3411911019765745, -
       1.0884675054377309, 1.416940735238965, 3.33569889674],
      [1.1276056488362338, -0.5184717355565056, -0.6714590697440526, -
       1.1406845870361446, 0.9257938051968776, 2.971788475193335],
      [-0.21202247769766364, 0.4951164758773363, -1.3121964545879525, -
       0.4806058935157951, 1.9781743709369604, 3.440733],
      [0.697515045436151, 0.40931074816764956, -1.1306045857368923, -
       0.09439293867058603, 1.9827951295482107, 3.764298132546647],
      [0.9496985991448699, -0.5406503069467344, -0.2009704561671725,
       0.5666888254433984, 0.42006316769378793, 3.550002194817929],
      [0.8439291161952339, 0.16532342049929666, -1.0083711505424846,
       0.3466158031218269, 1.023886763372576, 2.8198200726725]
    ])
  else:
    log.warn(f"The number of dof is not 6, no seed is provided!")
    return []
