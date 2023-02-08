import Network_Tables_Sender as nts
import tag

import cv2
import depthai as dai

#Special MOE one
import moe_apriltags as apriltag
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

debug = 1

if debug:
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)


buffer = []

#Creating pipeline
def create_pipeline():
    pipeline = dai.Pipeline()

    monoout = pipeline.createXLinkOut()
    monoout.setStreamName("mono")

    monocam = pipeline.createMonoCamera()
    monocam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monocam.setFps(60)
    monocam.setBoardSocket(dai.CameraBoardSocket.LEFT)

    monocam.out.link(monoout.input)
    return pipeline


detector = apriltag.Detector(families="tag16h5", nthreads=2)

class Transform:
    translation: np.ndarray
    rotation: R

    def __init__(self, translation: np.ndarray, rotation: R):
        if translation.shape != (3,):
            raise ValueError(f'Shape of translation is {translation.shape}')
        self.translation = translation
        self.rotation = rotation

    def inv(self) -> 'Transform':
        return Transform(
            rotation = self.rotation.inv(),
            translation = self.rotation.apply(-self.translation, inverse = True)
        )

    def combine(self, other: 'Transform') -> 'Transform':
        rotated = self.rotation.apply(other.translation)
            
        return Transform(
            rotation = self.rotation * other.rotation,
            translation = self.translation + rotated
        )
        #return R.concatenate([self, other])


camera_rs = Transform(
    translation = np.array([0,0,0]),
    rotation = R.identity(),
)


def calculate_pose(det: apriltag.Detection):
    #negate to account for rotation of tag
    tag_cs = Transform(
        translation=result.pose_t[:,0],
        rotation=R.from_matrix(-result.pose_R)
    )
    
    if debug:
        print(result.pose_R)

    cam_ts = tag_cs.inv()
    
    # print(translation_inv, tinv)

    tag_tl_fs = tag.tag_translation[det.tag_id]
    tag_ro_fs = tag.tag_rotation[det.tag_id]

    tag_fs = Transform(
        translation= tag_tl_fs,
        rotation=tag_ro_fs
    )

    cam_fs = tag_fs.combine(cam_ts) #Transforms camera in field space to tag in field space. Camera in robot space is then transformed into robot in camera space, which allows us to get robot in field space.
    robot_cs = camera_rs.inv()
    robot_fs = cam_fs.combine(robot_cs)

    return robot_fs


with dai.Device(create_pipeline()) as device:
    monoq = device.getOutputQueue(name="mono", maxSize=1, blocking=False)

    calibdata = device.readCalibration()
    intrinsics = calibdata.getDefaultIntrinsics(dai.CameraBoardSocket.LEFT)[0]
    oak_d_camera_params = (
        intrinsics[0][0],
        intrinsics[1][1],
        intrinsics[0][2],
        intrinsics[1][2],
    )


    while True:
        img = monoq.get().getCvFrame()

        results = detector.detect(img,
                                l=1,
                                r=8,
                                maxhamming=0,
                                estimate_tag_pose=True,
                                tag_size=.1524,
                                camera_params=oak_d_camera_params)
        
        if debug:
            cv2.imshow('foo', img)
            if cv2.waitKey(1) == ord('q'):
                break
        
        if len(results) == 0:
            continue
        
        results.sort(reverse=True, key = lambda x: x.pose_err)
        result: apriltag.Detection = results[0]

        rotation_cs = result.pose_R
        translation_cs = result.pose_t[:,0]

        robot_fs = calculate_pose(result)
            
        if(debug):
            if len(buffer) > 20:
                buffer = buffer[-20:]
            
            buffer.append([*robot_fs.translation, *translation_cs])
            b = np.array(buffer)
            ax1.cla()
            ax1.set(xlim=(-5,5), ylim=(-5,5))
            ax1.scatter([0], [0])
            ax1.plot(b[:,0], b[:,2])
            ax1.plot(b[:,3], b[:,5])
            plt.draw()
            plt.pause(.001)


        nts.send_pose(list(np.concatenate([robot_fs.translation, robot_fs.rotation.as_quat()]))) #Returns robot in field space.

        
