from .quaternion import Quaternion
from unittest import TestCase
import numpy as np


class QuaternionTest(TestCase):
	def test_identity(self):
		# Identity
		q1 = Quaternion.identity()
		self.assertEqual(1.0, q1.w)
		self.assertEqual(0.0, q1.x)
		self.assertEqual(0.0, q1.y)
		self.assertEqual(0.0, q1.z)

	def test_init_normalized(self):
		# Normalized
		q2 = Quaternion(0.5, 0.5, 0.5, 0.5)
		self.assertEqual(0.5, q2.w)
		self.assertEqual(0.5, q2.x)
		self.assertEqual(0.5, q2.y)
		self.assertEqual(0.5, q2.z)

	def test_init_unnormalized(self):
		# Unnormalized
		q3 = Quaternion(0.75, 0.3, 0.4, 0.5)
		self.assertEqual(0.75, q3.w)
		self.assertEqual(0.3, q3.x)
		self.assertEqual(0.4, q3.y)
		self.assertEqual(0.5, q3.z)

		q3 = q3.normalize()
		norm = np.sqrt(0.75 * 0.75 + 0.3 * 0.3 + 0.4 * 0.4 + 0.5 * 0.5)
		self.assertEqual(0.75 / norm, q3.w)
		self.assertEqual(0.3 / norm, q3.x)
		self.assertEqual(0.4 / norm, q3.y)
		self.assertEqual(0.5 / norm, q3.z)
		self.assertEqual(
			1.0,
			q3.w * q3.w
				+ q3.x * q3.x
				+ q3.y * q3.y
				+ q3.z * q3.z)
	
	def test_mul(self):
		# 90° CCW rotations around each axis
		c = np.cos(np.deg2rad(90.0) / 2.0)
		s = np.sin(np.deg2rad(90.0) / 2.0)
		xRot = Quaternion(c, s, 0.0, 0.0)
		yRot = Quaternion(c, 0.0, s, 0.0)
		zRot = Quaternion(c, 0.0, 0.0, s)

		# 90° CCW X rotation, 90° CCW Y rotation, and 90° CCW Z rotation should
		# produce a 90° CCW Y rotation
		expected = yRot
		actual = (zRot * yRot) * xRot
		self.assertAlmostEqual(expected.w, actual.w)
		self.assertAlmostEqual(expected.x, actual.x)
		self.assertAlmostEqual(expected.y, actual.y)
		self.assertAlmostEqual(expected.z, actual.z)
	
	def test_mul_identity(self):
		# Identity
		q = Quaternion(0.72760687510899891, 0.29104275004359953, 0.38805700005813276, 0.48507125007266594)
		actual = q * ~q
		self.assertEqual(actual, Quaternion.identity())
		self.assertEqual(1.0, actual.w)
		self.assertEqual(0.0, actual.x)
		self.assertEqual(0.0, actual.y)
		self.assertEqual(0.0, actual.z)

	def test_inverse(self):
		q = Quaternion(0.75, 0.3, 0.4, 0.5)
		inv = ~q

		self.assertEqual(q.w, inv.w)
		self.assertEqual(-q.x, inv.x)
		self.assertEqual(-q.y, inv.y)
		self.assertEqual(-q.z, inv.z)