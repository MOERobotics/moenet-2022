import tag
from .utils.geom.geom3 import Rotation3D, Transform3D, Pose3D, Translation3D
import numpy as np
import moe_apriltags as apriltag

class detection:
    #Tag detected
    tag_num : int

    #Tag position from robot
    relative_pose : Pose3D

    #Estimated field position
    absolute_pose: Pose3D

    #Error of measurement
    error : float

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

class TagDetector:
    def __init__(self, cam) -> None:
        self.detector = apriltag.Detector(families="tag16h5", nthreads=2)
        self.cam = cam

    def detect(self, cam) -> list[tuple[int, Transform3D]]:
        return []
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass