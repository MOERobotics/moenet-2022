from utils.debug import WebDebug, FieldId, RobotId, CameraId, TagId
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import numpy as np
from time import sleep

@dataclass
class Pose:
    translation: np.ndarray
    rotation: R

if __name__ == '__main__':
    debug = WebDebug()
    i = 0
    while True:
        with debug.frame() as frame:
            pose = Pose(
                translation=np.array([i * .01, 0., 0.], dtype=float),
                rotation=R.from_euler('xyz', [i * .0, 0, 0]),
            )

            frame.record(CameraId(0), TagId(1), pose)
        
        print('robot pose', i * .1)
        i += 1
        if i > 1e3:
            i = 0
        sleep(.01)