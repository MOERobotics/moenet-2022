from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import tag
import numpy as np

if TYPE_CHECKING:
    # Only load these modules if we need them
    import moe_apriltags as apriltag #Special MOE one
    import depthai as dai
import numpy as np

from utils.geom.geom3 import Rotation3D, Transform3D, Pose3D, Translation3D
from utils.debug import Debugger, DebugFrame, FieldId, RobotId, CameraId, TagId, WebDebug

debugger_type = 'web'
"""
You can set this variable to:
 - `None` -> no debugging
 - `"web"` -> runs web server
"""
simulate = False
"""
Set this to `True` if you want to simulate an apriltag rotating around, instead of using a camera
"""

camera0_rs = Transform3D(
    Translation3D(0,0,0),
    Rotation3D.from_rotation_matrix(np.array(
                                    [[ 0, 0, 1],
                                     [-1, 0, 0],
                                     [ 0,-1, 0]]
                                   ))
    # Rotation3D.identity()
    # + Rotation3D.from_axis_angle([1,0,0], 90, degrees=True)
    # + Rotation3D.from_axis_angle([0,1,0], 180, degrees=True)
    # + Rotation3D.from_axis_angle([1,0,0], 90, degrees=True)
)

class TagDetector:
    camera_id: int
    camera_rs: Transform3D
    def detect(self) -> list[tuple[int, Transform3D]]:
        return []
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

class OakTagDetector(TagDetector):
    @staticmethod
    def create_pipeline():
        import depthai as dai
        pipeline = dai.Pipeline()

        monoout = pipeline.createXLinkOut()
        monoout.setStreamName("mono")

        monocam = pipeline.createMonoCamera()
        monocam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monocam.setFps(30)
        monocam.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monocam.out.link(monoout.input)
        return pipeline
    
    def __init__(self) -> None:
        self.camera_id = 0
        self.camera_rs = camera0_rs

        self.detector = apriltag.Detector(families="tag16h5", nthreads=2)
    
    def __enter__(self):
        import depthai as dai
        self._device = dai.Device(self.create_pipeline())
        device: dai.Device = self._device.__enter__()
        self.monoq = device.getOutputQueue(name="mono", maxSize=1, blocking=False)
        calibdata = device.readCalibration()
        intrinsics = calibdata.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, destShape=(600,400))
        self._camera_params = (
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
        )
        return self
    
    def __exit__(self, *args):
        res = self._device.__exit__(*args)
        del self._device
        return res
    
    def detect(self) -> list[tuple[int, Transform3D]]:
        frame: 'dai.ImgFrame' = self.monoq.get()
        img: np.ndarray = frame.getCvFrame()

        if True:
            import cv2
            cv2.imshow('foo', img)
            if cv2.waitKey(1) == ord('q'):
                raise StopIteration
        
        detections: list[apriltag.Detection] = self.detector.detect(
            img,
            l=1,
            r=8,
            maxhamming=0,
            estimate_tag_pose=True,
            tag_size=.1524,
            camera_params=self._camera_params,
        )
        if len(detections) == 0:
            return []

        detections.sort(reverse=True, key = lambda x: x.pose_err)
        detection = detections[0]
        
        #rotation object is for the back of the apriltag
        tag_cs = Transform3D(
            Translation3D(detection.pose_t[:,0]),
            rotation=Rotation3D.from_rotation_matrix(detection.pose_R)
        )
        
        print(detection.pose_R)

        return [
            (detection.tag_id, tag_cs)
        ]


class FakeTagDetector(TagDetector):
    def __init__(self):
        self.camera_id = 0
        self.camera_rs = camera0_rs
        self.i = 0
    
    def detect(self) -> list[tuple[int, Transform3D]]:
        import time
        time.sleep(.01)
        if self.i < 60:
            ax = [1,0,0]
        elif self.i < 120:
            ax = [0,1,0]
        else:
            ax = [0,0,1]
        tag_cs = Transform3D(
            Translation3D(0,0,1),
            rotation=Rotation3D.from_axis_angle(ax, ((self.i % 60) - 30)*2, degrees=True),
        )
        self.i += 1
        self.i %= 180
        return [
            (1, tag_cs)
        ]


def robot_from_tag(tag_cs: Transform3D, tag_id: int, camera_rs: Transform3D, camera_id: int, dbf: Optional[DebugFrame] = None):
    """
    Compute the robot pose (`robot_fs`) from a tag detection (`tag_cs`)

    ## Parameters
    - `tag_cs` Detected tag pose, in camera space
    - `tag_id` AprilTag ID of detected tag
    - `camera_rs` Pose of camera that detected the tag, in robot space
    - `camera_id` ID of camera that detected the tag
    - `dbf` Debugging frame (optional)
    """

    #camera in tag space but using camera like axes, z axis is normal to tag
    camera_tcs = -tag_cs

    #----Works Up Till Here ------

    #tag in field space, this assumes x axis is normal to tag
    tag_fs = tag.tags[tag_id]

    #What tag camera space looks like in tag space
    tcs_ts : Pose3D = Pose3D(Translation3D(0,0,0),
                        Rotation3D.from_rotation_matrix(np.array(
                            [[0, 0,-1],
                             [1, 0, 0],
                             [0,-1, 0]]
                        )))

    #go from tag camera space to tag space
    camera_ts = tcs_ts.transform_by(camera_tcs)

    camera_fs = tag_fs.transform_by(camera_ts)

    robot_cs = -camera_rs
    
    # Translation works, but camera_rs having rotation is broken
    robot_fs = camera_fs.transform_by(robot_cs)

    robot_fs = camera_tcs

    if dbf is not None:
        fs = FieldId()
        rs = RobotId()
        cs = CameraId(camera_id)
        ts = TagId(tag_id)

        dbf.record(ts, cs, tag_cs)
        # dbf.record(rs, cs, robot_cs)

        dbf.record(cs, ts, camera_ts)
        # robot_ts = Pose3D.from_transform(robot_cs).transform_by(-camera_ts) # Good
        robot_ts = Pose3D.from_transform(camera_rs).relative_to(Pose3D.from_transform(tag_cs)) # Good
        dbf.record(rs, ts, robot_ts)

        dbf.record(cs, rs, camera_rs)
        tag_rs = Pose3D.from_transform(camera_ts).relative_to(Pose3D.from_transform(robot_cs))
        # tag_rs = -Transform3D(robot_ts.translation, robot_ts.rotation)
        dbf.record(ts, rs, tag_rs)
        
        dbf.record(ts, fs, tag_fs)
        dbf.record(cs, fs, camera_fs)
        dbf.record(rs, fs, robot_fs)

    return robot_fs


if __name__ == '__main__':
    import Network_Tables_Sender as nts
    import moe_apriltags as apriltag

    debugger: Debugger = WebDebug() if debugger_type == 'web' else Debugger()
    with (FakeTagDetector() if simulate else OakTagDetector()) as detector:
        print('ready')

        while True:
            detections = detector.detect()

            if len(detections) == 0:
                continue

            detection = detections[0]

            with debugger.frame() as dbf:
                for other_id, other_pose in detections[1:]:
                    dbf.record(TagId(other_id), CameraId(0), other_pose)
                
                robot_fs = robot_from_tag(detection[1], detection[0], detector.camera_rs, detector.camera_id, dbf)
            
            tl = robot_fs.translation
            q = robot_fs.rotation.to_quaternion()

            print(tl.x, tl.y, tl.z)

            pose = [tl.x, tl.y, tl.z, q.w, q.x, q.y, q.z]
            nts.send_pose(pose) #Returns robot in field space.
