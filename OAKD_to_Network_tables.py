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

@dataclass
class Transform:
    translation: np.ndarray
    rotation: R

    def inv(self) -> 'Transform':
        return Transform(
            rotation = self.rotation.inv(),
            translation = self.rotation.apply(-self.translation, inverse = True)
        )

    def combine(self, other: 'Transform') -> 'Transform':
        return Transform(
            rotation = R.concatenate(self, other),
            translation = self.translation + self.rotation.apply(other.translation)
        )
        #return R.concatenate([self, other])


robot_cam_tl = np.array([0,0,0])


def calculate_pose(det: apriltag.Detection):
    #negate to account for rotation of tag
    tag_cs = Transform(
        translation=result.pose_t[:,0],
        rotation=R.from_matrix(-result.pose_R)
    )
    
    if debug:
        print(result.pose_R)

    camera_ro_ts = tag_cs.rotation.inv()
    camera_tl_ts = tag_cs.rotation.apply(-tag_cs.translation, inverse=True)

    # print(translation_inv, tinv)

    tag_tl_fs = tag.tag_translation[det.tag_id]
    tag_ro_fs = tag.tag_rotation[det.tag_id]

    cam_tl_fs = tag_tl_fs + (tag_ro_fs.apply(camera_tl_ts))
    camera_ro_fs = R.concatenate([tag_ro_fs, camera_ro_ts])

    return camera_ro_ts, camera_tl_ts


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

        rotation_ts, translation_ts = calculate_pose(result)
            
        if(debug):
            if len(buffer) > 20:
                buffer = buffer[-20:]
            
            buffer.append([*translation_ts, *translation_cs])
            b = np.array(buffer)
            ax1.cla()
            ax1.set(xlim=(-5,5), ylim=(-5,5))
            ax1.scatter([0], [0])
            ax1.plot(b[:,0], b[:,2])
            ax1.plot(b[:,3], b[:,5])
            plt.draw()
            plt.pause(.001)
        
        #Original frame of reference: x - side to side, y - up and down, z - towards target
        #New frame of reference: x - towards target, y - side to side, z - up and down
        translation_ts = translation_ts[[2,0,1]]

        translation_ts[0] = (-1)**(tag.tag_rotation[result.tag_id])* \
                                translation_ts[0] + \
                                tag.posvec[result.tag_id][0]
        
        translation_ts[1] = (-1)**(tag.tag_rotation[result.tag_id]^1)* \
                                translation_ts[1] + \
                                tag.posvec[result.tag_id][1]


        translation_ts[2] += tag.posvec[result.tag_id][2]


        #Rotation details
        uvecp = [0,0,1] #plane vector
        uvecn = [0,1,0] #normal vector
        rotvec = rotation_ts@uvecp
        rollvec = rotation_ts@uvecn

        #Original frame of reference: x - side to side, y - up and down, z - towards target
        #New frame of reference: x - towards target, y - side to side, z - up and down
        rotvec = rotvec[[2,0,1]]
        rollvec = rollvec[[2,0,1]]

        #All angles given in deg, +- 180

        #yaw - counterclockwise - 0 in line with [1,0,0]
        yaw = np.arctan2(rotvec[1], rotvec[0])

        #pitch - counterclockwise - 0 in line with [1,0,0]
        pitch = np.arctan2(rotvec[2], rotvec[0])

        #roll - counterclockwise - 0 in line with [0,0,1]
        roll = np.arctan2(rollvec[1], rollvec[2])

        #compile angles and turn them into degrees
        angles = [yaw, pitch, roll]
        angles = [np.rad2deg(a) for a in angles]

        nts.send_pose([*translation_ts, *angles])