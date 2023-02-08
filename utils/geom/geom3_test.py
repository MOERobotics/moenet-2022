from unittest import TestCase
import numpy as np
import numpy.testing as npt
from .geom2 import Rotation2D, Translation2D, Transform2D, Twist2D, Pose2D
from .geom3 import Rotation3D, Translation3D, Transform3D, Twist3D, Pose3D

xAxis = np.array([1, 0, 0], dtype=float)
yAxis = np.array([0, 1, 0], dtype=float)
zAxis = np.array([0, 0, 1], dtype=float)

class Rotation3DTest(TestCase):
    def test_axisangle_vs_euler(self):
        rot1 = Rotation3D.from_axis_angle(xAxis, np.pi / 3)
        rot2 = Rotation3D.from_euler(np.pi / 3, 0, 0)
        self.assertEqual(rot1, rot2)

        rot3 = Rotation3D.from_axis_angle(yAxis, np.pi / 3)
        rot4 = Rotation3D.from_euler(0, np.pi / 3, 0)
        self.assertEqual(rot3, rot4)

        
        rot5 = Rotation3D.from_axis_angle(zAxis, np.pi / 3)
        rot6 = Rotation3D.from_euler(0, 0, np.pi / 3)
        self.assertEqual(rot5, rot6)
  
    def test_from_rotation_matrix_identity(self):
        # No rotation
        actual = Rotation3D.from_rotation_matrix(np.identity(3))
        expected = Rotation3D.identity()
        self.assertEqual(expected, actual)

    def test_rotation_matrix_z(self):
        # 90 degree CCW rotation around z-axis
        actual = Rotation3D.from_rotation_matrix([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])
        expected = Rotation3D.from_euler(0, 0, 90, degrees=True)
        self.assertEqual(expected, actual)

    def test_rotation_matrix_invalid(self):
        # Matrix that isn't orthogonal
        R3 = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])
        with self.assertRaises(ValueError):
            Rotation3D(R3)

        # Matrix that's orthogonal but not special orthogonal
        R4 = np.identity(3, dtype=float) * 2.
        with self.assertRaises(ValueError):
            Rotation3D(R4)
  
    def test_between(self):
        # 90 degree CW rotation around y-axis
        rot1 = Rotation3D.between(xAxis, zAxis)
        expected1 = Rotation3D.from_axis_angle(yAxis, -np.pi / 2)
        self.assertEqual(expected1, rot1)

        # 45 degree CCW rotation around z-axis
        rot2 = Rotation3D.between(xAxis, np.array([1, 1, 0]))
        expected2 = Rotation3D.from_axis_angle(zAxis, np.pi / 4)
        self.assertEqual(expected2, rot2)

        # 0 degree rotation of x-axes
        rot3 = Rotation3D.between(xAxis, xAxis)
        self.assertEqual(Rotation3D.identity(), rot3)

        # 0 degree rotation of y-axes
        rot4 = Rotation3D.between(yAxis, yAxis)
        self.assertEqual(Rotation3D.identity(), rot4)

        # 0 degree rotation of z-axes
        rot5 = Rotation3D.between(zAxis, zAxis)
        self.assertEqual(Rotation3D.identity(), rot5)

        # 180 degree rotation tests. For 180 degree rotations, any quaternion with
        # an orthogonal rotation axis is acceptable. The rotation axis and initial
        # vector are orthogonal if their dot product is zero.

        # 180 degree rotation of x-axes
        rot6 = Rotation3D.between(xAxis, xAxis * -1)
        q6 = rot6.to_quaternion()
        self.assertEqual(0, q6.w)
        self.assertEqual(
            0,
            q6.x * xAxis[0] + q6.y * xAxis[1] + q6.z * xAxis[2])

        # 180 degree rotation of y-axes
        rot7 = Rotation3D.between(yAxis, yAxis * -1)
        q7 = rot7.to_quaternion()
        self.assertEqual(0, q7.w)
        self.assertEqual(
            0,
            q7.x * yAxis[0] + q7.y * yAxis[1] + q7.z * yAxis[2])

        # 180 degree rotation of z-axes
        rot8 = Rotation3D.between(zAxis, zAxis * -1)
        q8 = rot8.to_quaternion()
        self.assertEqual(0, q8.w)
        self.assertEqual(
            0,
            q8.x * zAxis[0] + q8.y * zAxis[1] + q8.z * zAxis[2])

  
    def test_rad2deg(self):
        rot1 = Rotation3D.from_axis_angle(zAxis, np.pi / 3)
        rot1b = Rotation3D.from_axis_angle(zAxis, 60, degrees=True)
        self.assertEqual(rot1, rot1b)
        self.assertAlmostEqual(np.deg2rad(0), rot1.x)
        self.assertAlmostEqual(np.deg2rad(0), rot1.y)
        self.assertAlmostEqual(np.deg2rad(60), rot1.z)

        rot2 = Rotation3D.from_axis_angle(zAxis, np.pi / 4)
        rot2b = Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        self.assertEqual(rot2, rot2b)
        self.assertAlmostEqual(np.deg2rad(0), rot2.x)
        self.assertAlmostEqual(np.deg2rad(0), rot2.y)
        self.assertAlmostEqual(np.deg2rad(45), rot2.z)

  
    def test_rad_and_deg(self):
        rot1 = Rotation3D.from_axis_angle(zAxis, np.deg2rad(45))
        rot1b = Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        self.assertEqual(rot1, rot1b)
        
        self.assertAlmostEqual(0, rot1.x)
        self.assertAlmostEqual(0, rot1.y)
        self.assertAlmostEqual(np.pi / 4, rot1.z)

        rot2 = Rotation3D.from_axis_angle(zAxis, np.deg2rad(30))
        rot2b = Rotation3D.from_axis_angle(zAxis, 30, degrees=True)
        self.assertEqual(rot2, rot2b)
        
        self.assertAlmostEqual(0, rot2.x)
        self.assertAlmostEqual(0, rot2.y)
        self.assertAlmostEqual(np.pi / 6, rot2.z)
  
    def test_rotation_loop(self):
        rot = Rotation3D.identity()

        # Rotate 90º around X
        rot += Rotation3D.from_euler(90, 0, 0, degrees=True)
        expected = Rotation3D.from_euler(90, 0, 0, degrees=True)
        self.assertEqual(expected, rot)

        # Rotate 90º around Y
        rot += Rotation3D.from_euler(0, 90, 0, degrees=True)
        expected = Rotation3D.from_axis_angle(
            np.array([1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]),
            120,
            degrees=True
        )
        self.assertEqual(expected, rot)

        # Rotate 90º around Z
        rot += Rotation3D.from_euler(0, 0, 90, degrees=True)
        expected = Rotation3D.from_euler(0, 90, 0, degrees=True)
        self.assertEqual(expected, rot)

        # Rotate back 90º around Y
        rot += Rotation3D.from_euler(0, -90, 0, degrees=True)
        self.assertEqual(Rotation3D.identity(), rot)

    def test_rotate_from_zero_x(self):
        zero = Rotation3D.identity()
        rotated = zero.rotate_by(Rotation3D.from_axis_angle(xAxis, 90, degrees=True))

        expected = Rotation3D.from_axis_angle(xAxis, 90, degrees=True)
        self.assertEqual(expected, rotated)
  
    def test_rotate_from_zero_y(self):
        zero = Rotation3D.identity()
        rotated = zero.rotate_by(Rotation3D.from_axis_angle(yAxis, 90, degrees=True))

        expected = Rotation3D.from_axis_angle(yAxis, 90, degrees=True)
        self.assertEqual(expected, rotated)
  
    def test_rotate_from_zero_z(self):
        zero = Rotation3D.identity()
        rotated = zero.rotate_by(Rotation3D.from_axis_angle(zAxis, 90, degrees=True))

        expected = Rotation3D.from_axis_angle(zAxis, 90, degrees=True)
        self.assertEqual(expected, rotated)
  
    def test_rotate_nonzero_x(self):
        rot = Rotation3D.from_axis_angle(xAxis, 90, degrees=True)
        rot = rot + Rotation3D.from_axis_angle(xAxis, 30, degrees=True)

        expected = Rotation3D.from_axis_angle(xAxis, 120, degrees=True)
        self.assertEqual(expected, rot)
  
    def test_scale(self):
        rot = Rotation3D.from_axis_angle(xAxis, 10, degrees=True)

        scaled = rot * 4
        summed = rot + rot + rot + rot
        expected = Rotation3D.from_axis_angle(xAxis, 40, degrees=True)

        self.assertEqual(scaled, expected)
        self.assertEqual(summed, expected)
    
    def test_rotate_nonzero_y(self):
        rot = Rotation3D.from_axis_angle(yAxis, 90, degrees=True)
        rot = rot + Rotation3D.from_axis_angle(yAxis, 30, degrees=True)

        expected = Rotation3D.from_axis_angle(yAxis, 120, degrees=True)
        self.assertEqual(expected, rot)
  
    def test_rotate_nonzero_z(self):
        rot = Rotation3D.from_axis_angle(zAxis, 90, degrees=True)
        rot = rot + Rotation3D.from_axis_angle(zAxis, 30, degrees=True)

        expected = Rotation3D.from_axis_angle(zAxis, 120, degrees=True)
        self.assertEqual(expected, rot)
  
    def test_sub(self):
        rot1 = Rotation3D.from_axis_angle(zAxis, 70, degrees=True)
        rot2 = Rotation3D.from_axis_angle(zAxis, 30, degrees=True)

        self.assertAlmostEqual((rot1 - rot2).z, np.deg2rad(40))

  
    def test_eq(self):
        rot1 = Rotation3D.from_axis_angle(zAxis, 43, degrees=True)
        rot2 = Rotation3D.from_axis_angle(zAxis, 43, degrees=True)
        self.assertEqual(rot1, rot2)

        rot1 = Rotation3D.from_axis_angle(zAxis, -180, degrees=True)
        rot2 = Rotation3D.from_axis_angle(zAxis, 180, degrees=True)
        self.assertEqual(rot1, rot2)

  
    def test_axis_angle_x(self):
        rot = Rotation3D.from_axis_angle(xAxis, 90, degrees=True)
        axis, angle = rot.to_axis_angle()
        npt.assert_array_equal(axis, xAxis)
        self.assertAlmostEqual(np.pi / 2, angle)

    def test_axis_angle_y(self):
        rot = Rotation3D.from_axis_angle(yAxis, 45, degrees=True)
        axis, angle = rot.to_axis_angle()
        npt.assert_array_equal(axis, yAxis)
        self.assertAlmostEqual(np.pi / 4, angle)

    def test_axis_angle_z(self):
        rot = Rotation3D.from_axis_angle(zAxis, 60, degrees=True)
        axis, angle = rot.to_axis_angle()
        npt.assert_array_equal(axis, zAxis)
        self.assertAlmostEqual(np.pi / 3, angle)
  
    def test_2d(self):
        rotation = Rotation3D.from_euler(
            20,
            30,
            40,
            degrees=True
        )
        expected = Rotation2D.from_degrees(40)

        self.assertEqual(expected, rotation.to_2d())
  
    def test_neq(self):
        rot1 = Rotation3D.from_axis_angle(zAxis, 43.0, degrees=True)
        rot2 = Rotation3D.from_axis_angle(zAxis, 43.5, degrees=True)
        self.assertNotEqual(rot1, rot2)
  
    def test_interpolate_1(self):
        # 50 + (70 - 50) * 0.5 = 60
        rot1 = Rotation3D.from_axis_angle(xAxis, 50, degrees=True)
        rot2 = Rotation3D.from_axis_angle(xAxis, 70, degrees=True)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(np.deg2rad(60), interpolated.x)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.y)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.z)
    
    def test_interpolate_2(self):
        # -160 minus half distance between 170 and -160 (15) = -175
        rot1 = Rotation3D.from_axis_angle(xAxis, 170, degrees=True)
        rot2 = Rotation3D.from_axis_angle(xAxis, -160, degrees=True)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(np.deg2rad(-175), interpolated.x)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.y)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.z)
    
    def test_interpolate_3(self):
        # 50 + (70 - 50) * 0.5 = 60
        rot1 = Rotation3D.from_axis_angle(yAxis, 50, degrees=True)
        rot2 = Rotation3D.from_axis_angle(yAxis, 70, degrees=True)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.x)
        self.assertAlmostEqual(np.deg2rad(60), interpolated.y)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.z)
    
    def test_interpolate_4(self):
        # -160 minus half distance between 170 and -160 (165) = 5
        rot1 = Rotation3D.from_axis_angle(yAxis, 170, degrees=True)
        rot2 = Rotation3D.from_axis_angle(yAxis, -160, degrees=True)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(np.deg2rad(180), interpolated.x)
        self.assertAlmostEqual(np.deg2rad(-5), interpolated.y)
        self.assertAlmostEqual(np.deg2rad(180), interpolated.z)
    
    def test_interpolate_5(self):
        # 50 + (70 - 50) * 0.5 = 60
        rot1 = Rotation3D.from_axis_angle(zAxis, 50, degrees=True)
        rot2 = Rotation3D.from_axis_angle(zAxis, 70, degrees=True)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.x)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.y)
        self.assertAlmostEqual(np.deg2rad(60), interpolated.z)
    
    def test_interpolate_6(self):
        # -160 minus half distance between 170 and -160 (15) = -175
        rot1 = Rotation3D.from_axis_angle(zAxis, 170, degrees=True)
        rot2 = Rotation3D.from_axis_angle(zAxis, -160, degrees=True)
        interpolated = rot1.interpolate(rot2, 0.5)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.x)
        self.assertAlmostEqual(np.deg2rad(0), interpolated.y)
        self.assertAlmostEqual(np.deg2rad(-175), interpolated.z)
    
    def test_inv_identity(self):
        I = Rotation3D.identity()
        self.assertEqual(I, -I)


class Translation3DTest(TestCase):
    def test_add(self):
        one = Translation3D(1, 3, 5)
        two = Translation3D(2, 5, 8)

        sum = one + two
        self.assertAlmostEqual(3, sum.x)
        self.assertAlmostEqual(8, sum.y)
        self.assertAlmostEqual(13, sum.z)
    
    def test_sub(self):
        one = Translation3D(1, 3, 5)
        two = Translation3D(2, 5, 8)

        difference = one - two
        self.assertAlmostEqual(-1, difference.x)
        self.assertAlmostEqual(-2, difference.y)
        self.assertAlmostEqual(-3, difference.z)

    def test_rotate(self):
        translation = Translation3D(1, 2, 3)

        rotated1 = translation.rotate_by(Rotation3D.from_axis_angle(xAxis, 90, degrees=True))
        self.assertAlmostEqual(1, rotated1.x)
        self.assertAlmostEqual(-3, rotated1.y)
        self.assertAlmostEqual(2, rotated1.z)

        rotated2 = translation.rotate_by(Rotation3D.from_axis_angle(yAxis, 90, degrees=True))
        self.assertAlmostEqual(3, rotated2.x)
        self.assertAlmostEqual(2, rotated2.y)
        self.assertAlmostEqual(-1, rotated2.z)

        rotated3 = translation.rotate_by(Rotation3D.from_axis_angle(zAxis, 90, degrees=True))
        self.assertAlmostEqual(-2, rotated3.x)
        self.assertAlmostEqual(1, rotated3.y)
        self.assertAlmostEqual(3, rotated3.z)

    def test_2d(self):
        translation = Translation3D(1, 2, 3)
        expected = Translation2D(1, 2)

        self.assertEqual(expected, translation.as_2d())

    def test_mul(self):
        original = Translation3D(3, 5, 7)
        mult = original * 3
        self.assertAlmostEqual(9, mult.x)
        self.assertAlmostEqual(15, mult.y)
        self.assertAlmostEqual(21, mult.z)
    
    def test_div(self):
        original = Translation3D(3, 5, 7)
        div = original / 2
        self.assertAlmostEqual(1.5, div.x)
        self.assertAlmostEqual(2.5, div.y)
        self.assertAlmostEqual(3.5, div.z)

    def test_norm(self):
        one = Translation3D(3, 5, 7)
        self.assertAlmostEqual(np.sqrt(83), one.norm())

    def test_distance(self):
        one = Translation3D(1, 1, 1)
        two = Translation3D(6, 6, 6)
        self.assertAlmostEqual(5 * np.sqrt(3), one.distance_to(two))

    def test_inv(self):
        original = Translation3D(-4.5, 7, 9)
        inverted = -original
        self.assertAlmostEqual(4.5, inverted.x)
        self.assertAlmostEqual(-7, inverted.y)
        self.assertAlmostEqual(-9, inverted.z)

    def test_eq(self):
        one = Translation3D(9, 5.5, 3.5)
        two = Translation3D(9, 5.5, 3.5)
        self.assertEqual(one, two)

    def test_neq(self):
        one = Translation3D(9, 5.5, 3.5)
        two = Translation3D(9, 5.7, 3.5)
        self.assertNotEqual(one, two)
    
    def test_polar(self):
        one = Translation3D.from_polar(np.sqrt(2), Rotation3D.from_axis_angle(zAxis, 45, degrees=True))
        self.assertAlmostEqual(1, one.x)
        self.assertAlmostEqual(1, one.y)
        self.assertAlmostEqual(0, one.z)

        two = Translation3D.from_polar(2, Rotation3D.from_axis_angle(zAxis, 60, degrees=True))
        self.assertAlmostEqual(1, two.x)
        self.assertAlmostEqual(np.sqrt(3), two.y)
        self.assertAlmostEqual(0, two.z)


class Transform3DTest(TestCase):
    def test_inv_identity(self):
        I = Transform3D.identity()
        self.assertEqual(I, -I)
    
    def test_inv(self):
        initial = Pose3D(
            Translation3D(1, 2, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )
        transform = Transform3D(
            Translation3D(5, 0, 0),
            Rotation3D.from_axis_angle(zAxis, 5, degrees=True)
        )

        transformed = initial + transform
        untransformed = transformed + (-transform)

        self.assertAlmostEqual(initial.x, untransformed.x)
        self.assertAlmostEqual(initial.y, untransformed.y)
        self.assertAlmostEqual(initial.z, untransformed.z)
        self.assertAlmostEqual(initial.rotation.z, untransformed.rotation.z)

    def test_composition(self):
        initial = Pose3D(
            Translation3D(1, 2, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )
        transform1 = Transform3D(
            Translation3D(5, 0, 0),
            Rotation3D.from_axis_angle(zAxis, 5, degrees=True)
        )
        transform2 = Transform3D(
            Translation3D(0, 2, 0),
            Rotation3D.from_axis_angle(zAxis, 5, degrees=True)
        )

        transformedSeparate = (initial + transform1) + transform2
        transformedCombined = initial + (transform1 + transform2)

        self.assertAlmostEqual(transformedSeparate.x, transformedCombined.x)
        self.assertAlmostEqual(transformedSeparate.y, transformedCombined.y)
        self.assertAlmostEqual(transformedSeparate.z, transformedCombined.z)
        self.assertEqual(transformedSeparate.rotation, transformedCombined.rotation)
    
    def test_composition2(self):
        initial = Pose3D(
            Translation3D(1, 0,0),
            Rotation3D.from_axis_angle(zAxis, 90, degrees=True)
        )
        transform1 = Transform3D(
            Translation3D(1,0,0),
            Rotation3D.from_axis_angle(xAxis, 90, degrees=True),
        )
        transform2 = Transform3D(
            Translation3D(1,0,0),
            Rotation3D.from_axis_angle(yAxis, 90, degrees=True)
        )

        transformedSeparate = (initial + transform1) + transform2
        transformedCombined = initial + (transform1 + transform2)

        self.assertEqual(transformedSeparate.rotation, transformedCombined.rotation)
        self.assertEqual(transformedSeparate.translation, transformedCombined.translation)


class Twist3DTest(TestCase):
    def test_straight_x(self):
        straight = Twist3D([5, 0, 0], [0, 0, 0])
        straightPose = Pose3D.zero().exp(straight)

        expected = Pose3D(
            Translation3D(5, 0, 0),
            Rotation3D.identity()
        )
        self.assertEqual(expected, straightPose)
    
    def test_straight_y(self):
        straight = Twist3D([0, 5, 0], [0, 0, 0])
        straightPose = Pose3D.zero().exp(straight)

        expected = Pose3D(
            Translation3D(0, 5, 0),
            Rotation3D.identity()
        )
        self.assertEqual(expected, straightPose)
    
    def test_straight_z(self):
        straight = Twist3D([0, 0, 5], [0, 0, 0])
        straightPose = Pose3D.zero().exp(straight)

        expected = Pose3D(
            Translation3D(0, 0, 5),
            Rotation3D.identity()
        )
        self.assertEqual(expected, straightPose)
    
    def test_quarter_cirle(self):
        quarterCircle = Twist3D([5 / 2 * np.pi, 0, 0], [0, 0, np.pi / 2])
        quarterCirclePose = Pose3D.zero().exp(quarterCircle)

        expected = Pose3D(
            Translation3D(5, 5, 0),
            Rotation3D.from_axis_angle(zAxis, 90, degrees=True)
        )
        self.assertEqual(expected, quarterCirclePose)
    
    def test_diagonal_noDtheta(self):
        diagonal = Twist3D([2, 2, 0], [0, 0, 0])
        diagonalPose = Pose3D.zero().exp(diagonal)

        expected = Pose3D(
            Translation3D(2, 2, 0),
            Rotation3D.identity()
        )
        self.assertEqual(expected, diagonalPose)
    
    def test_eq(self):
        one = Twist3D([5, 1, 0], [0, 0, 3])
        two = Twist3D([5, 1, 0], [0, 0, 3])
        self.assertEqual(one, two)
    
    def test_neq(self):
        one = Twist3D([5, 1, 0], [0, 0, 3])
        two = Twist3D([5, 1.2, 0], [0, 0, 3])
        self.assertNotEqual(one, two)
    
    def test_log_x(self):
        start = Pose3D.zero()
        end = Pose3D(
            Translation3D(0, 5, 5),
            Rotation3D.from_euler(90, 0, 0, degrees=True)
        )

        twist = start.log(end)

        expected = Twist3D([0, (5/2) * np.pi, 0], [np.deg2rad(90), 0, 0])
        print(twist)
        print(expected)
        self.assertEqual(expected, twist)

        # Make sure computed twist gives back original end pose
        reapplied = start.exp(twist)
        self.assertEqual(end, reapplied)

    def test_log_y(self):
        start = Pose3D.zero()
        end = Pose3D(
            Translation3D(5, 0, 5),
            Rotation3D.from_euler(0, 90, 0, degrees=True))

        twist = start.log(end)

        expected = Twist3D([0, 0, 5 / 2 * np.pi], [0, np.pi / 2, 0])
        self.assertEqual(expected, twist)

        # Make sure computed twist gives back original end pose
        reapplied = start.exp(twist)
        self.assertEqual(end, reapplied)

    def test_log_z(self):
        start = Pose3D.zero()
        end = Pose3D(
            Translation3D(5, 5, 0),
            Rotation3D.from_euler(0, 0, 90, degrees=True)
        )

        twist = start.log(end)

        expected = Twist3D([5 / 2 * np.pi, 0, 0], [0, 0, np.pi / 2])
        self.assertEqual(expected, twist)

        # Make sure computed twist gives back original end pose
        reapplied = start.exp(twist)
        self.assertEqual(end, reapplied)


class Pose3DTest(TestCase):
    def test_transform(self):
        

        initial = Pose3D(
            Translation3D(1, 2, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )
        transformation = Transform3D(
            Translation3D(5, 0, 0),
            Rotation3D.from_axis_angle(zAxis, 5, degrees=True)
        )

        transformed = initial + (transformation)
        self.assertAlmostEqual(1 + 5 / np.sqrt(2), transformed.x)
        self.assertAlmostEqual(2 + 5 / np.sqrt(2), transformed.y)
        self.assertAlmostEqual(np.deg2rad(50), transformed.rotation.z)

    def test_relative(self):
        initial = Pose3D(
            Translation3D(0, 0, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )
        last = Pose3D(
            Translation3D(5, 5, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )

        finalRelativeToInitial = last.relative_to(initial)
        self.assertAlmostEqual(5 * np.sqrt(2), finalRelativeToInitial.x)
        self.assertAlmostEqual(0, finalRelativeToInitial.y)
        self.assertAlmostEqual(0, finalRelativeToInitial.rotation.z)

    def test_eq(self):
        one = Pose3D(
            Translation3D(0, 5, 0),
            Rotation3D.from_axis_angle(zAxis, 43, degrees=True)
        )
        two = Pose3D(
            Translation3D(0, 5, 0),
            Rotation3D.from_axis_angle(zAxis, 43, degrees=True)
        )
        self.assertEqual(one, two)

    def test_neq(self):
        one = Pose3D(
            Translation3D(0, 5, 0),
            Rotation3D.from_axis_angle(zAxis, 43, degrees=True)
        )
        two = Pose3D(
            Translation3D(0, 1.524, 0),
            Rotation3D.from_axis_angle(zAxis, 43, degrees=True)
        )
        self.assertNotEqual(one, two)
    
    def test_sub(self):
        initial = Pose3D(
            Translation3D(0, 0, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )
        last = Pose3D(
            Translation3D(5, 5, 0),
            Rotation3D.from_axis_angle(zAxis, 45, degrees=True)
        )

        transform = last - initial
        self.assertAlmostEqual(5 * np.sqrt(2), transform.x)
        self.assertAlmostEqual(0, transform.y)
        self.assertAlmostEqual(0, transform.rotation.z)
    
    def test_2d(self):
        pose = Pose3D(
            Translation3D(1, 2, 3),
            Rotation3D.from_euler(
                20,
                30,
                40,
                degrees=True,
            )
        )
        expected = Pose2D(Translation2D(1, 2), Rotation2D.from_degrees(40))

        self.assertEqual(expected, pose.as_2d())