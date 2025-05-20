#! /usr/bin/env python
import unittest
import numpy as np
# Tested package
from data_types.baldor import constants as br_constants
from data_types.baldor import euler as br_euler


class TestModule(unittest.TestCase):
  def test_axes_sequences(self):
    for axes in br_constants._AXES2TUPLE.keys():
      ai, aj, ak = np.random.sample(3) * 2 * np.pi
      # Axis-angle
      axis, angle = br_euler.to_axis_angle(ai, aj, ak, axes)
      # Transform
      br_euler.to_transform(ai, aj, ak, axes)  # T =

  def test_to_axis_angle(self):
    axis, angle = br_euler.to_axis_angle(0, 1.5, 0, 'szyx')
    np.testing.assert_allclose(axis, [0, 1, 0])
    np.testing.assert_almost_equal(angle, 1.5)

  def test_to_quaternion(self):
    q = br_euler.to_quaternion(1, 2, 3, 'ryxz')
    expected = [0.43595284, 0.31062245, -0.71828702, 0.44443511]
    np.testing.assert_allclose(q, expected)
    q = br_euler.to_quaternion(1, 2, 3, (2, 0, 0, 1))
    # Test parity
    q = br_euler.to_quaternion(1, 2, 3, 'rzyz')
    expected = [-0.2248451, 0.70807342, 0.45464871, 0.4912955]
    np.testing.assert_allclose(q, expected)

  def test_to_transform(self):
    T = br_euler.to_transform(1, 2, 3, 'syxz')
    np.testing.assert_almost_equal(np.sum(T[0]), -1.34786452)
    T = br_euler.to_transform(1, 2, 3, (0, 1, 0, 1))
    np.testing.assert_almost_equal(np.sum(T[0]), -0.383436184)
