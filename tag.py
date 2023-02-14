import json
from utils.geom.geom3 import Pose3D, Translation3D, Rotation3D
from utils.geom.quaternion import Quaternion

tagf = open('tags.json')

data = json.load(tagf)

def parse_tag(tag):
    pose = tag['pose']
    posvec = [float(pos) for pos in pose['translation'].values()]

    translation = Translation3D(
        pose['translation']['x'],
        pose['translation']['y'],
        pose['translation']['z'],
    )

    quat = pose['rotation']['quaternion']
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

tags = {
    tag['ID']: parse_tag(tag)
    for tag in data['tags']
}