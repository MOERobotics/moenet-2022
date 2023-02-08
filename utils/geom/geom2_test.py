import unittest
import numpy as np
from .geom2 import Rotation2D, Translation2D, Transform2D, Twist2D, Pose2D

class Rotation2DTest(unittest.TestCase):
    def test_rad2deg(self):
        rot1 = Rotation2D.from_raidans(np.pi / 3)
        self.assertAlmostEqual(60, rot1.to_degrees())

        rot2 = Rotation2D.from_raidans(np.pi / 4)
        self.assertAlmostEqual(45, rot2.to_degrees())
    
    def test_deg2rad(self):
        rot1 = Rotation2D.from_degrees(45)
        self.assertAlmostEqual(np.pi / 4, rot1.to_radians())

        rot2 = Rotation2D.from_degrees(30)
        self.assertAlmostEqual(np.pi / 6, rot2.to_radians())
    
    def test_rotate_zero(self):
        zero = Rotation2D.identity()
        rotated = zero.rotate_by(Rotation2D.from_degrees(90))

        self.assertAlmostEqual(np.pi / 2, rotated.to_radians())
        self.assertAlmostEqual(90, rotated.to_degrees())
    
    def test_rotate_nonzero(self):
        rot = Rotation2D.from_degrees(90)
        rot = rot + Rotation2D.from_degrees(30)

        self.assertAlmostEqual(120, rot.to_degrees())
    
    def test_minus(self):
        rot1 = Rotation2D.from_degrees(70)
        rot2 = Rotation2D.from_degrees(30)

        self.assertAlmostEqual(40, (rot1 - rot2).to_degrees())
    
    def test_eq(self):
        rot1 = Rotation2D.from_degrees(43)
        rot2 = Rotation2D.from_degrees(43)

        self.assertEqual(rot1, rot2)

        rot1 = Rotation2D.from_degrees(-180)
        rot2 = Rotation2D.from_degrees(+180)

        self.assertEqual(rot1, rot2)
    
    def test_neq(self):
        rot1 = Rotation2D.from_degrees(43.0)
        rot2 = Rotation2D.from_degrees(43.5)

        self.assertNotEqual(rot1, rot2)
    
    def test_interpolate(self):
        # 50 + (70 - 50) * 0.5 = 60
        rot1 = Rotation2D.from_degrees(50)
        rot2 = Rotation2D.from_degrees(70)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(60., interpolated.to_degrees())

        # -160 minus half distance between 170 and -160 (15) = -175
        rot1 = Rotation2D.from_degrees(170)
        rot2 = Rotation2D.from_degrees(-160)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(-175, interpolated.to_degrees())


class Translation2DTest(unittest.TestCase):
    def test_sum(self):
        one = Translation2D(1, 3)
        two = Translation2D(2, 5)

        sum = one + two

        self.assertAlmostEqual(3, sum.x)
        self.assertAlmostEqual(8, sum.y)

    def test_difference(self):
        one = Translation2D(1, 3)
        two = Translation2D(2, 5)

        difference = one - two

        self.assertAlmostEqual(-1, difference.x)
        self.assertAlmostEqual(-2, difference.y)

    def test_rotate(self):
        another = Translation2D(3, 0)
        rotated = another.rotate_by(Rotation2D.from_degrees(90))

        self.assertAlmostEqual(0, rotated.x)
        self.assertAlmostEqual(3, rotated.y)
    
    def test_mul(self):
        original = Translation2D(3, 5)
        res = original * 3.

        self.assertAlmostEqual( 9, res.x)
        self.assertAlmostEqual(15, res.y)

    def test_div(self):
        original = Translation2D(3, 5)
        res = original / 2

        self.assertAlmostEqual(1.5, res.x)
        self.assertAlmostEqual(2.5, res.y)

    def test_norm(self):
        one = Translation2D(3, 4)
        self.assertAlmostEqual(5, one.norm())

    def test_distance(self):
        one = Translation2D(1, 1)
        two = Translation2D(6, 6)

        self.assertAlmostEqual(5 * np.sqrt(2), one.distance_to(two))
    
    def test_inverse(self):
        original = Translation2D(-4.5, 7)
        inverted = -original

        self.assertAlmostEqual(4.5, inverted.x)
        self.assertAlmostEqual(-7, inverted.y)

    def test_eq(self):
        one = Translation2D(9, 5.5)
        two = Translation2D(9, 5.5)
        self.assertEqual(one, two)

    def test_new(self):
        one = Translation2D(9, 5.5)
        two = Translation2D(9, 5.7)
        self.assertNotEqual(one, two)
    
    def test_polar(self):
        one = Translation2D.from_polar(np.sqrt(2), Rotation2D.from_degrees(45))
        self.assertAlmostEqual(1, one.x)
        self.assertAlmostEqual(1, one.y)

        two = Translation2D.from_polar(2, Rotation2D.from_degrees(60))
        self.assertAlmostEqual(1, two.x)
        self.assertAlmostEqual(np.sqrt(3), two.y)


class Transform2DTest(unittest.TestCase):
    def test_inv(self):
        initial = Pose2D(Translation2D(1, 2), Rotation2D.from_degrees(45))
        transform = Transform2D(Translation2D(5, 0), Rotation2D.from_degrees(5))

        transformed = initial + transform
        untransformed = transformed + (-transform)

        self.assertAlmostEqual(initial.x, untransformed.x)
        self.assertAlmostEqual(initial.y, untransformed.y)
        self.assertAlmostEqual(initial.rotation.to_degrees(), untransformed.rotation.to_degrees())

    def test_composition(self):
        initial = Pose2D(Translation2D(1, 2), Rotation2D.from_degrees(45))
        transform1 = Transform2D(Translation2D(5, 0), Rotation2D.from_degrees(5))
        transform2 = Transform2D(Translation2D(0, 2), Rotation2D.from_degrees(5))

        transformedSeparate = (initial + transform1) + transform2
        transformedCombined = initial + (transform1 + transform2)

        self.assertAlmostEqual(transformedSeparate.x, transformedCombined.x)
        self.assertAlmostEqual(transformedSeparate.y, transformedCombined.y)
        self.assertAlmostEqual(transformedSeparate.rotation.to_degrees(), transformedCombined.rotation.to_degrees())


class Twist2DTest(unittest.TestCase):
    def test_straight(self):
        straight = Twist2D(5, 0, 0)
        straightPose = Pose2D.zero().exp(straight)

        expected = Pose2D(Translation2D(5, 0), Rotation2D(0))
        self.assertEqual(expected, straightPose)

    def test_quarter_cirlcle(self):
        quarterCircle = Twist2D(5 / 2 * np.pi, 0, np.pi / 2)
        quarterCirclePose = Pose2D.zero().exp(quarterCircle)

        expected = Pose2D(Translation2D(5, 5), Rotation2D.from_degrees(90))
        self.assertEqual(expected, quarterCirclePose)

    def test_diagonal_no_dtheta(self):
        diagonal = Twist2D(2, 2, 0)
        diagonalPose = Pose2D.zero().exp(diagonal)

        expected = Pose2D(Translation2D(2, 2), Rotation2D.identity())
        self.assertEqual(expected, diagonalPose)

    def test_eq(self):
        one = Twist2D(5, 1, 3)
        two = Twist2D(5, 1, 3)
        self.assertEqual(one, two)

    def test_neq(self):
        one = Twist2D(5, 1, 3)
        two = Twist2D(5, 1.2, 3)
        self.assertNotEqual(one, two)

    def test_pose_log(self):
        start = Pose2D.zero()
        end = Pose2D(Translation2D(5, 5), Rotation2D.from_degrees(90))

        twist = start.log(end)

        expected = Twist2D(5 / 2 * np.pi, 0, np.pi / 2)
        self.assertEqual(expected, twist)

        # Make sure computed twist gives back original end pose
        reapplied = start.exp(twist)
        self.assertEqual(end, reapplied)


class Pose2DTest(unittest.TestCase):
    def test_transform(self):
        initial = Pose2D(Translation2D(1, 2), Rotation2D.from_degrees(45))
        transformation = Transform2D(Translation2D(5, 0), Rotation2D.from_degrees(5))

        transformed = initial + transformation

        self.assertAlmostEqual(1 + 5 / np.sqrt(2), transformed.x)
        self.assertAlmostEqual(2 + 5 / np.sqrt(2), transformed.y)
        self.assertAlmostEqual(50, transformed.rotation.to_degrees())

    def test_relative(self):
        initial = Pose2D(Translation2D(0, 0), Rotation2D.from_degrees(45))
        last    = Pose2D(Translation2D(5, 5), Rotation2D.from_degrees(45))

        finalRelativeToInitial = last.relative_to(initial)

        self.assertAlmostEqual(5 * np.sqrt(2), finalRelativeToInitial.x)
        self.assertAlmostEqual(0, finalRelativeToInitial.y)
        self.assertAlmostEqual(0, finalRelativeToInitial.rotation.to_degrees())

    def test_eq(self):
        one = Pose2D(Translation2D(0, 5), Rotation2D.from_degrees(43))
        two = Pose2D(Translation2D(0, 5), Rotation2D.from_degrees(43))
        self.assertEqual(one, two)

    def test_neq(self):
        one = Pose2D(Translation2D(0, 5), Rotation2D.from_degrees(43))
        two = Pose2D(Translation2D(0, 1.524), Rotation2D.from_degrees(43))
        self.assertNotEqual(one, two)

    def test_sub(self):
        initial = Pose2D(Translation2D(0, 0), Rotation2D.from_degrees(45))
        last    = Pose2D(Translation2D(5, 5), Rotation2D.from_degrees(45))

        transform = last - initial

        self.assertAlmostEqual(5 * np.sqrt(2), transform.x)
        self.assertAlmostEqual(0, transform.y)
        self.assertAlmostEqual(0, transform.rotation.to_degrees())