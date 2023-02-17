from unittest import TestCase
from OAKD_to_Network_tables import robot_from_tag, ItemDetection, estimate_pose
from utils.geom import Pose3D, Transform3D, Translation3D, Rotation3D
from utils.debug import DebugFrame, CameraId, FieldId, RobotId, TagId
import numpy as np
import numpy.testing as npt
from tag import tags

class RobotPoseTest(TestCase):
	def test_cts1(self):
		tag_cs = Transform3D(
			Translation3D(0,0,1),
			Rotation3D.from_axis_angle('y', 90, degrees=True)
		)
		camera_ts = tag_cs.inv()
		print(camera_ts.rotation.to_euler(degrees=True))
		self.assertAlmostEqual(camera_ts.x, 1)
		self.assertAlmostEqual(camera_ts.y, 0)
		self.assertAlmostEqual(camera_ts.z, 0)
	
	def test_cts2(self):
		tag_cs = Transform3D(
			Translation3D(0,0,1),
			Rotation3D.from_axis_angle('y', 180, degrees=True)
		)
		camera_ts = -tag_cs
		print(camera_ts.rotation.to_euler(degrees=True))
		self.assertAlmostEqual(camera_ts.x, 0)
		self.assertAlmostEqual(camera_ts.y, 0)
		self.assertAlmostEqual(camera_ts.z, 1)
	
	def test_rfs(self):
		dbf = DebugFrame()
		tag_cs = Transform3D(
			Translation3D(0,0,1),
			Rotation3D.from_axis_angle('y', 180, degrees=True)
		)
		camera_rs = Transform3D(
			Translation3D(0,1,0),
			Rotation3D.identity(),
		)

		robot_fs = robot_from_tag(tag_cs, 1, camera_rs, 0, dbf)

		cam_ts = dbf.get_pose(CameraId(0), TagId(1))
		self.assertAlmostEqual(cam_ts.x, 0)
		self.assertAlmostEqual(cam_ts.y, 0)
		self.assertAlmostEqual(cam_ts.z, 1)
	
	def test_mat(self):
		camera_ts = Transform3D(
			Translation3D(3,3,3),
			Rotation3D.from_euler(1,2,3),
			# Rotation3D.identity()
		)
		tag_fs = Pose3D(
			Translation3D(1,1,1),
			Rotation3D.identity()
		)

		def compute_wpilib(camera_ts: Transform3D, tag_fs:Pose3D):
			tcs_ts = Transform3D(
				Translation3D(0,0,0),
				Rotation3D.from_rotation_matrix([
					[0, 0,-1],
					[1, 0, 0],
					[0,-1, 0]
				]))
			robot_fs = tag_fs.transform_by(tcs_ts).transform_by(camera_ts)
			return robot_fs.translation.as_vec()

		def compute_mat(camera_ts: Transform3D, tag_fs: Pose3D):
			translation_ts = camera_ts.translation.as_vec()
			rotation_ts = camera_ts.rotation.to_matrix()

			tag2field = np.array([
				[0,0,1],
				[-1,0,0],
				[0,-1,0]
			])
			if tag_fs.rotation.to_quaternion().w:
				tag2field = tag2field@np.array([[-1,0,0],[0,1,0],[0,0,-1]])
			
			translation_fs = tag2field@translation_ts
			translation_fs += tag_fs.translation.as_vec()

			#Rotation details
			uvecp = [0,0,1] #plane vector
			uvecn = [0,-1,0] #normal vector
			rotvec = tag2field@(rotation_ts@uvecp)
			rollvec = tag2field@(rotation_ts@uvecn)

			#yaw - counterclockwise - 0 in line with [1,0,0]
			yaw = np.arctan2(rotvec[1], rotvec[0])

			#pitch - counterclockwise - 0 in line with [1,0,0]
			pitch = np.arctan2(rotvec[2], rotvec[0])

			#roll - counterclockwise - 0 in line with [0,0,1]
			roll = np.arctan2(rollvec[1], rollvec[2])
			return translation_fs

		r_wpilib = compute_wpilib(camera_ts, tag_fs)
		r_mat = compute_mat(camera_ts, tag_fs)
		npt.assert_almost_equal(r_wpilib, r_mat)

	def test_pose(self):
		tags = {
			0: Pose3D(
				Translation3D(3,3,3),
				Rotation3D.identity(),
			),
			1: Pose3D(
				Translation3D(5,5,5),
				Rotation3D.identity(),
			)
		}

		cameras = {
			CameraId(0): Transform3D.identity()
		}

		detections = [
			ItemDetection(
				camera_id=0,
				item_id=TagId(0),
				pose_cs=Transform3D(
					Translation3D(1,2,3),
					Rotation3D.from_euler(1,2,3),
				),
				ambiguity=0.7,
			),
			ItemDetection(
				camera_id=0,
				item_id=TagId(1),
				pose_cs=Transform3D(
					Translation3D(4,2,3),
					Rotation3D.identity(),
				),
				ambiguity=0.3
			),
			ItemDetection(
				camera_id=0,
				item_id=TagId(0),
				pose_cs=Transform3D(
					Translation3D(1,2,3),
					Rotation3D.from_euler(1,2,3),
				),
				ambiguity=0.4
			)
		]

		pose = estimate_pose(detections, cameras, tags=tags)
		self.assertAlmostEqual(1, pose.x)
		self.assertAlmostEqual(3, pose.y)
		self.assertAlmostEqual(2, pose.z)