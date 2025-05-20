#! /usr/bin/env python
import unittest
import numpy as np
# Tested package
from data_types.baldor import axis_angle as br_axis_angle
from data_types.baldor import transform as br_transform
from data_types.baldor import quaternion as br_quaternion
from data_types.baldor import euler as br_euler
from data_types.baldor import vector as br_vector
from data_types.baldor import constants as br_constants


class TestModule(unittest.TestCase):
  def test_are_equal(self):
    T0 = np.diag([1, 1, 1, 1])
    T1 = np.eye(4)
    self.assertTrue(br_quaternion.are_equal(T0, T1))
    T2 = br_quaternion.to_transform([1, 0, 0, 0])
    T3 = br_quaternion.to_transform([-1, 0, 0, 0])
    self.assertTrue(br_quaternion.are_equal(T2, T3))

  def test_between_axes(self):
    # Random axis / angle
    np.random.seed(123)
    axis = br_vector.unit(np.random.randn(3))
    angle = np.deg2rad(45)
    transform = br_axis_angle.to_transform(axis, angle)
    newaxis = np.dot(transform[:3, :3], br_constants.Z_AXIS)
    est_transform = br_transform.between_axes(br_constants.Z_AXIS, newaxis)
    np.testing.assert_allclose(transform[:3, 2], est_transform[:3, 2])
    # Edge case 1
    newaxis = -br_constants.Z_AXIS
    transform = br_transform.between_axes(br_constants.Z_AXIS, newaxis)
    _, angle, _ = br_transform.to_axis_angle(transform)
    np.testing.assert_allclose(angle, np.pi)
    # Edge case 2
    newaxis = br_constants.Z_AXIS
    transform = br_transform.between_axes(-br_constants.Z_AXIS, newaxis)
    _, angle, _ = br_transform.to_axis_angle(transform)
    np.testing.assert_allclose(angle, np.pi)

  def test_inverse(self):
    q = br_quaternion.random()
    T = br_quaternion.to_transform(q)
    Tinv = br_transform.inverse(T)
    expected_zero = np.dot(T, Tinv) - np.eye(4)
    assert np.all(np.abs(expected_zero[:])) < 1e-6

  def test_random(self):
    T = br_transform.random()
    Tinv = br_transform.inverse(T)
    expected_zero = np.dot(T, Tinv) - np.eye(4)
    assert np.all(np.abs(expected_zero[:])) < 1e-6

  def test_to_axis_angle(self):
    axis = np.random.sample(3) - 0.5
    angle = (np.random.sample(1) - 0.5) * (2 * np.pi)
    point = np.random.sample(3) - 0.5
    T0 = br_axis_angle.to_transform(axis, angle, point)
    axis, angle, point = br_transform.to_axis_angle(T0)
    T1 = br_axis_angle.to_transform(axis, angle, point)
    self.assertTrue(br_transform.are_equal(T0, T1))
    # The eigenvector of the rotation matrix corresponding to eigenvalue 1 is
    # the axis of rotation

    def rotation_eigenvalue_test():
      T2 = np.eye(4)
      T2[:3, :3] *= 2
      br_transform.to_axis_angle(T2)
    self.assertRaises(ValueError, rotation_eigenvalue_test)
    # Test simple axes
    T = br_axis_angle.to_transform(br_constants.X_AXIS, angle)
    v0, angle, _ = br_transform.to_axis_angle(T)
    np.testing.assert_allclose(v0, br_constants.X_AXIS, atol=1e-8)
    T = br_axis_angle.to_transform(br_constants.Y_AXIS, angle)
    v1, angle, _ = br_transform.to_axis_angle(T)
    np.testing.assert_allclose(v1, br_constants.Y_AXIS, atol=1e-8)
    T = br_axis_angle.to_transform(br_constants.Z_AXIS, angle)
    v2, angle, _ = br_transform.to_axis_angle(T)
    np.testing.assert_allclose(v2, br_constants.Z_AXIS, atol=1e-8)

  def test_to_dual_quaternion(self):
    qr0 = br_quaternion.random()
    qt0 = br_quaternion.random()
    # Check transform consistency
    T0 = br_quaternion.dual_to_transform(qr0, qt0)
    Tinv = br_transform.inverse(T0)
    np.testing.assert_allclose(np.dot(T0, Tinv), np.eye(4), rtol=1e-5, atol=1e-8)
    # Convert back
    qr1, qt1 = br_transform.to_dual_quaternion(T0)
    T1 = br_quaternion.dual_to_transform(qr1, qt1)
    # T0 and T1 must be equal
    self.assertTrue(br_transform.are_equal(T0, T1))

  def test_to_euler(self):
    # Test all tuples few times
    for axes in br_constants._AXES2TUPLE.keys():
      ai, aj, ak = np.random.sample(3) * 2 * np.pi
      T0 = br_euler.to_transform(ai, aj, ak, axes)
      al, be, ga = br_transform.to_euler(T0, axes)
      T1 = br_euler.to_transform(al, be, ga, axes)
      np.testing.assert_allclose(T0, T1)

  def test_to_quaternion(self):
    # Identity test
    q = br_transform.to_quaternion(np.identity(4), isprecise=True)
    np.testing.assert_allclose(q, [1, 0, 0, 0])
    # Simple rotation test
    q = br_transform.to_quaternion(np.diag([1, -1, -1, 1]))
    self.assertTrue(np.allclose(q, [0, 1, 0, 0]) or
                    np.allclose(q, [0, -1, 0, 0]))
    # Dummy test
    T0 = br_axis_angle.to_transform((1, 2, 3), 0.123)
    q = br_transform.to_quaternion(T0, True)
    np.testing.assert_allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786],
                               rtol=1e-05, atol=1e-8)
    # Permutation matrix
    T1 = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]]
    br_transform.to_quaternion(T1, True)  # q1 =
    # Round trip
    np.random.seed(123)   # Generate always the same transforms
    for _ in range(100):
      T0 = br_transform.random()
      T0[:3, 3] = 0
      isprecise = np.random.rand() > 0.5
      q0 = br_transform.to_quaternion(T0, isprecise)
      T1 = br_quaternion.to_transform(q0)
      self.assertTrue(br_transform.are_equal(T0, T1))
