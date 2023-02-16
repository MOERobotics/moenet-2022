from typing import TypedDict
import json
from utils.geom.geom3 import Pose3D, Translation3D, Rotation3D
from utils.geom.quaternion import Quaternion

# Type definitions
class JSONFieldDimensions(TypedDict):
    length: float
    width: float

class JSONTranslation3D(TypedDict):
    x: float
    y: float
    z: float

class JSONQuaternion(TypedDict):
    W: float
    X: float
    Y: float
    Z: float

class JSONRotation3D(TypedDict):
    quaternion: JSONQuaternion

class JSONPose3D(TypedDict):
    translation: JSONTranslation3D
    rotation: JSONRotation3D

class JSONPose3D(TypedDict):
    translation: JSONTranslation3D
    rotation: JSONRotation3D

class JSONTagInfo(TypedDict):
    ID: int
    pose: JSONPose3D

class JSONAprilTagField(TypedDict):
    tags: list[JSONTagInfo]
    field: JSONFieldDimensions

def parse_tag(tag: JSONTagInfo):
    pose_raw = tag['pose']
    translation_raw = pose_raw['translation']

    translation = Translation3D(
        translation_raw['x'],
        translation_raw['y'],
        translation_raw['z'],
    )

    quat = pose_raw['rotation']['quaternion']
    rotation = Rotation3D(
        Quaternion(
            quat['W'],
            quat['X'],
            quat['Y'],
            quat['Z'],
        )
    )
    return Pose3D(
        translation,
        rotation,
    )


def load_field(field_path: str):
    with open(field_path, 'r') as f:
        tag_data: JSONAprilTagField = json.load(f)
    
    return {
        tag['ID']: parse_tag(tag)
        for tag in tag_data['tags']
    }


tags = load_field('./data/field-2023-chargedup.json')