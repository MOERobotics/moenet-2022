from unittest import TestCase
from OAKD_to_Network_tables import robot_from_tag
from utils.geom import Pose3D, Transform3D, Translation3D, Rotation3D
from utils.debug import DebugFrame, CameraId, FieldId, RobotId, TagId

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
        camera_ts = tag_cs.inv()
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