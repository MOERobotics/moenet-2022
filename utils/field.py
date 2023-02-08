from typing import Optional, List
from .geom.geom3 import Pose3D
from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class AprilTag:
    id: int
    pose: Pose3D
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, AprilTag) and (self.id == other.id) and (self.pose == other.pose)
    
    def __hash__(self) -> int:
        return hash((self.id, self.pose))

class FieldDimensions:
    pass

class AprilTagFieldLayout:
    @staticmethod
    def from_json(path: Path):
        pass
    
    def __init__(self, tags: List[AprilTag], field_dimensions: FieldDimensions) -> None:
        pass