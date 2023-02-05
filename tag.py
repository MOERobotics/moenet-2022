import json
import numpy as np
from scipy.spatial.transform import Rotation as R

tagf = open('tags.json')

data = json.load(tagf)

tag_translation = [0]*9
tag_rotation = [0]*9

for tag in data["tags"]:
    pose = tag['pose']
    posvec = [float(pos) for pos in pose['translation'].values()]
    tag_translation[tag['ID']] = np.array(posvec)

    rotation = pose["rotation"]['quaternion'] 
    tag_rotation[tag['ID']] = R.from_quat([rotation['W'], rotation['X'], rotation['Y'], rotation['Z']])
