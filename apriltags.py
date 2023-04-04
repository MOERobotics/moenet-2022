import tag
from utils.geom.geom3 import Rotation3D, Transform3D, Pose3D, Translation3D
import numpy as np
import moe_apriltags as apriltag

class detection:
    #Tag detected
    tag_id : int

    #Tag position from robot in robot space
    relative_pose : Transform3D

    #Estimated field position of robot
    absolute_pose: Pose3D

    #Error of measurement
    error : float

    @classmethod
    def convert(cls, cdetection : apriltag.Detection):
        converted = detection()
        converted.tag_id = cdetection.tag_id
        converted.relative_pose = Transform3D(
            Translation3D(cdetection.pose_t[:,0]),
            rotation=Rotation3D.from_rotation_matrix(cdetection.pose_R)
        )
        return converted

class estimated_pose:
    absolute_pose : Pose3D

    def __init__(self, pose : Pose3D) -> None:
        absolute_pose = pose

def combine_detections(detections: list[detection]) -> estimated_pose:
    #sort by pose error
    if len(detections):
        #Remove error portion
        comb_detection : detection = min(detections, key = lambda x : x.error)
        position : estimated_pose = estimated_pose(comb_detection)
        return position
    else:
        return None

def network_format(position : estimated_pose) -> list[float]:
    if position is None:
        return []
    return [*position.absolute_pose.translation.as_vec(),
            *position.absolute_pose.rotation.to_quaternion()]

import depthai as dai
class TagDetector:
    def __init__(self, cam) -> None:
        self.detector = apriltag.Detector(families="tag16h5", nthreads=2)
        self.cam = cam
        self.monoq = cam.device.getOutputQueue(name="mono", maxSize=1, blocking=False)
        calibdata =  cam.device.readCalibration()
        intrinsics = calibdata.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, destShape=(3*cam.resolution//2,cam.resolution))
        self._camera_params = (
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
        )

    def detect(self) -> list[detection]:
        frame: 'dai.ImgFrame' = self.monoq.get()
        img: np.ndarray = frame.getCvFrame()
        detections: list[apriltag.Detection] = self.detector.detect(
            img,
            l=1,
            r=8,
            maxhamming=0,
            estimate_tag_pose=True,
            tag_size=.1524,
            camera_params=self._camera_params,
        )

        final_detections = []
        for tag in detections:
            print(tag)
            converted : detection = detection.convert(tag)
            tag_cs = detection.relative_pose

            #Camera in robot space
            cam_robot = self.cam.pose

            #tag camera space - z axis is in the normal direction from robot space
            tagcs_robot = cam_robot.transform_by(tag_cs)

            #What tag camera space looks like in tag space
            tagcs_ts : Pose3D = Pose3D(Translation3D(0,0,0),
                                Rotation3D.from_rotation_matrix(np.array(
                                    [[0, 0,-1],
                                    [1, 0, 0],
                                    [0,-1, 0]]
                                )))

            #What tag space looks like in tag camera space
            tag_tagcs = -tagcs_ts

            tag_robot = tagcs_robot.transform_by(tag_tagcs)

            #tag position in field space
            tag_fs = tag.tags[converted.tag_id]

            #robot position in field space
            robot_fs = tag_fs.transform_by(-tag_robot)

            converted.relative_pose = tag_robot
            converted.absolute_pose = robot_fs

            final_detections.append(converted)
        return final_detections    
    
    def __enter__(self):
        
        return self

    def __exit__(self, *args):
        pass