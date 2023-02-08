from typing import Optional, TYPE_CHECKING
import tag

import depthai as dai

#Special MOE one
if TYPE_CHECKING:
    import moe_apriltags as apriltag
import numpy as np

from scipy.spatial.transform import Rotation as R
from utils.debug import Debugger, WebDebug, DebugFrame, FieldId, RobotId, CameraId, TagId

def create_pipeline():
    pipeline = dai.Pipeline()

    monoout = pipeline.createXLinkOut()
    monoout.setStreamName("mono")

    monocam = pipeline.createMonoCamera()
    monocam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monocam.setFps(60)
    monocam.setBoardSocket(dai.CameraBoardSocket.LEFT)

    monocam.out.link(monoout.input)
    return pipeline

class Transform:
    translation: np.ndarray
    rotation: R

    def __init__(self, translation: np.ndarray, rotation: R):
        if translation.shape != (3,):
            raise ValueError(f'Shape of translation is {translation.shape}')
        self.translation = translation
        self.rotation = rotation

    def inv(self) -> 'Transform':
        return Transform(
            rotation = self.rotation.inv(),
            translation = self.rotation.apply(-self.translation, inverse = True)
        )

    def combine(self, other: 'Transform') -> 'Transform':
        rotated = self.rotation.apply(other.translation)
        if rotated.shape != (3, ):
            rotated = rotated[0]
            
        return Transform(
            rotation = self.rotation * other.rotation,
            translation = self.translation + rotated
        )


camera_rs = Transform(
    translation = np.array([0,0,0]),
    rotation = R.identity(),
)


def calculate_pose(det: apriltag.Detection, dbf: Optional[DebugFrame] = None):
    tag_cs = Transform(
        translation=result.pose_t[:,0],
        rotation=R.from_matrix(-result.pose_R)
    )

    cam_ts = tag_cs.inv()

    tag_tl_fs = tag.tag_translation[det.tag_id]
    tag_ro_fs = tag.tag_rotation[det.tag_id]

    tag_fs = Transform(
        translation= tag_tl_fs,
        rotation=tag_ro_fs
    )

    cam_fs = tag_fs.combine(cam_ts) #Transforms camera in field space to tag in field space. Camera in robot space is then transformed into robot in camera space, which allows us to get robot in field space.
    robot_cs = camera_rs.inv()
    robot_fs = cam_fs.combine(robot_cs)

    if dbf is not None:
        fs = FieldId()
        rs = RobotId()
        cs = CameraId(0)
        ts = TagId(det.tag_id)
        dbf.record(ts, cs, tag_cs)
        dbf.record(cs, ts, cam_ts)
        dbf.record(ts, fs, tag_fs)
        dbf.record(cs, fs, cam_fs)
        dbf.record(cs, rs, camera_rs)
        dbf.record(rs, cs, robot_cs)
        dbf.record(rs, fs, robot_fs)

    return robot_fs


if __name__ == '__main__':
    import Network_Tables_Sender as nts
    import moe_apriltags as apriltag

    debugger: Debugger = WebDebug()

    detector = apriltag.Detector(families="tag16h5", nthreads=2)

    with dai.Device(create_pipeline()) as device:
        device: dai.Device
        monoq = device.getOutputQueue(name="mono", maxSize=1, blocking=False)

        calibdata = device.readCalibration()
        intrinsics = calibdata.getDefaultIntrinsics(dai.CameraBoardSocket.LEFT)[0]
        oak_d_camera_params = (
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
        )

        print('ready')

        while True:
            img = monoq.get().getCvFrame()
            results: list[apriltag.Detection] = detector.detect(
                img,
                l=1,
                r=8,
                maxhamming=0,
                estimate_tag_pose=True,
                tag_size=.1524,
                camera_params=oak_d_camera_params
            )

            with debugger.frame() as dbf:
                if dbf is not None:
                    import cv2
                    cv2.imshow('foo', img)
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                if len(results) == 0:
                    continue
                
                results.sort(reverse=True, key = lambda x: x.pose_err)
                result = results[0]

                rotation_cs = result.pose_R
                translation_cs = result.pose_t[:,0]

                robot_fs = calculate_pose(result, dbf)
            

            pose = [*robot_fs.translation, *robot_fs.rotation.as_quat()]
            nts.send_pose(pose) #Returns robot in field space.
