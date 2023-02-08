from typing import Union, Literal, Optional, ContextManager, Protocol
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation as R

class FieldId:
    def __init__(self):
        self.type = 'field'
class RobotId:
    def __init__(self):
        self.type = 'robot'
class TagId:
    def __init__(self, id: int):
        self.type = 'tag'
        self.id = id
class CameraId:
    def __init__(self, id: int):
        self.type = 'camera'
        self.id = id

ReferenceFrame = Union[FieldId, RobotId, TagId, CameraId]
ItemId = Union[RobotId, TagId, CameraId]

class PoseLike(Protocol):
    translation: np.ndarray[Literal[3], float]
    rotation: R

class DebugFrame:
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
        translation = pose.translation
        rotation = pose.rotation.as_quat()
        poses = self._records.setdefault(item, dict())
        poses[reference_frame] = (translation, rotation)
        

class Debugger:
    
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