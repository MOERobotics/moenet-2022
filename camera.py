import os
from multiprocessing import Queue
from enum import Enum
from utils.geom.geom3 import Rotation3D, Transform3D, Pose3D, Translation3D

class Camera:
    mxid: str
    type = Enum('type', {'default':None, 'S2':'S2', 'Lite':'Lite',})
    mode = Enum('mode', {'default':None, 'tag':'tag', 'hybrid': 'hybrid',})
    pose: Pose3D = Pose3D.zero()


def camera_init(info_file: str, apriltag_queue: Queue, object_queue: Queue):
    cam = Camera()
    import json
    unpacked_json = open(info_file)
    camera_data = json.load(unpacked_json)
    
    #Set up
    if True:
        cam.mxid = camera_data['MXID']
        cam.type = camera_data['Type']
        cam.mode = camera_data['Mode']

        cam.pose = Pose3D(Translation3D(camera_data['Pose']['Translation'].values()),
                        Rotation3D(camera_data['Pose']['Rotation'].values()))
    
        

    