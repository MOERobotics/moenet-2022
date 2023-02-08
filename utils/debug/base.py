from typing import Union, Literal, Optional, ContextManager, Protocol, TYPE_CHECKING
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation as R
if TYPE_CHECKING:
    from ..geom.geom3 import Rotation3D, Translation3D

class FieldId:
    "Field reference frame"
    def __init__(self):
        self.type = 'field'
class RobotId:
    "Robot object id/reference frame"
    def __init__(self):
        self.type = 'robot'
class TagId:
    "Tag object id/reference frame"
    def __init__(self, id: int):
        self.type = 'tag'
        self.id = id
class CameraId:
    "Camera object id/reference frame"
    def __init__(self, id: int):
        self.type = 'camera'
        self.id = id

ReferenceFrame = Union[FieldId, RobotId, TagId, CameraId]
"Reference frames ids"

ItemId = Union[RobotId, TagId, CameraId]
"Game item ids"

class PoseLike(Protocol):
    "Abstraction of a pose for debugging"

    translation: Union[np.ndarray[Literal[3], float], 'Translation3D']
    rotation: Union[R, 'Rotation3D']


class DebugFrame:
    "Records the poses seen (in different reference frames) for debugging"

    _records: dict[ItemId, dict[ReferenceFrame, tuple[np.ndarray, np.ndarray]]]
    def __init__(self):
        self._records = dict()
    
    def __setitem__(self, key: tuple[ItemId, ReferenceFrame], pose: PoseLike):
        item, frame = key
        self.record(item, frame, pose)
    
    def __getitem__(self, key: tuple[ItemId, ReferenceFrame]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        item, frame = key
        poses = self._records.get(item, None)
        if poses is None:
            return None
        return poses.get(frame, None)

    def record(self, item: ItemId, reference_frame: ReferenceFrame, pose: PoseLike):
        "Record pose"
        if isinstance(pose.translation, np.ndarray):
            translation =  pose.translation
        else:
            translation = pose.translation.as_vec()
        
        if isinstance(pose.rotation, R):
            rotation = pose.rotation.as_quat()[[1,2,3,0]]
        else:
            q = pose.rotation.to_quaternion()
            rotation = [
                q.w,
                q.x,
                q.y,
                q.z
            ]
        poses = self._records.setdefault(item, dict())
        poses[reference_frame] = (translation, rotation)
        

class Debugger:
    "No-op debugger base class"
    def frame(self) -> ContextManager[Optional[DebugFrame]]:
        @contextmanager
        def helper():
            frame = DebugFrame()
            try:
                yield frame
            finally:
                self.finish_frame(frame)
        return helper()
    
    def finish_frame(self, frame: DebugFrame):
        pass