#! /usr/bin/env python
import unittest
import numpy as np
# Tested package
from data_types.baldor import vector as br_vector
from data_types.baldor import quaternion as br_quaternion
from data_types.baldor import constants as br_constants


class TestModule(unittest.TestCase):
  def test_unit(self):
    v = np.random.sample(3)
    u = br_vector.unit(v)
    np.testing.assert_allclose(np.linalg.norm(u), 1, atol=1e-8)

  def test_norm(self):
    v = np.random.sample(3)
    np.testing.assert_allclose(br_vector.norm(v), np.linalg.norm(v), atol=1e-8)

  def test_perpendicular(self):
    u = np.random.sample(3)
    v = br_vector.perpendicular(u)
    np.testing.assert_allclose(np.dot(u, v), 0, atol=1e-8)
    # Test zero vector

    def zero_vector_test():
      br_vector.perpendicular(np.zeros(3))
    self.assertRaises(ValueError, zero_vector_test)
    # Test Z_AXIS case
    u1 = br_constants.Z_AXIS
    v1 = br_vector.perpendicular(u1)
    np.testing.assert_allclose(np.dot(u1, v1), 0, atol=1e-8)

  def test_skew(self):
    vector = np.random.sample(3)
    R = br_vector.skew(vector)
    np.testing.assert_allclose(-R, R.T)

  def test_transform_between_vectors(self):
    u0 = br_vector.unit(np.random.sample(3))
    q = br_quaternion.random()
    T0 = br_quaternion.to_transform(q)
    v0 = br_vector.unit(np.dot(T0, np.hstack((u0, 1)))[:3])
    T1 = br_vector.transform_between_vectors(u0, v0)
    v1 = br_vector.unit(np.dot(T1, np.hstack((u0, 1)))[:3])
    np.testing.assert_allclose(v0, v1, atol=1e-8)
    # Test same vector
    T2 = br_vector.transform_between_vectors(u0, u0)
    expected_zero = T2 - np.eye(4)
    assert np.all(np.abs(expected_zero[:])) < 1e-6
