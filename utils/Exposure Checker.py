import cv2
import depthai as dai
import moe_apriltags as apriltag
from time import monotonic
import keyboard

exposure_time = 350 #exposure time in us
increment = 50 #Increment that the exposure time can be increased/decreased by

#OAK-D

#Camera Parameters (fx, fy, cx, cy)
oak_d_camera_params = (794.311279296875, 794.311279296875, 620.0385131835938, 371.195068359375)


#Creating pipeline
pipeline = dai.Pipeline()

monocam = pipeline.create(dai.node.MonoCamera)

monoout = pipeline.create(dai.node.XLinkOut)

monoout.setStreamName("mono")

monocam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monocam.setFps(30)
monocam.setBoardSocket(dai.CameraBoardSocket.LEFT)

monocam.initialControl.setManualExposure(exposure_time,800)

monocam.out.link(monoout.input)

#Set up dynamic control
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(monocam.inputControl)


detector = apriltag.Detector(families="tag16h5", nthreads=2)

cnt = 0

ct = monotonic()
lastlook = ct

buffer_size = 100
times = [0]*buffer_size

with dai.Device(pipeline) as device:
    
    
#    calibdata = device.readCalibration()
#    print(calibdata.getDefaultIntrinsics(dai.CameraBoardSocket.LEFT))
    
    #Set up Queues
    monoq = device.getOutputQueue(name = "mono")
    controlQueue = device.getInputQueue(controlIn.getStreamName())
    
    #Set up 
    cq = [0]*buffer_size
    ptr1 = 0
    ptr2 = 0
    print('Ready')
    while True:
        #timing data
        ct = monotonic()
        img = monoq.get().getCvFrame()


        #Count Frames
        good = 0
        
        #Use this to convert to grayscale if needed - should be reading from the gray camera from oak-D
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = detector.detect(img, estimate_tag_pose=True, tag_size=.1524, camera_params=oak_d_camera_params)
        # print(results)
        for result in results:
            good = 1
        
        if good:
            cnt += 1
            cq[ptr2] = ct
            ptr2 += 1
            if(ptr2 >= buffer_size):
                ptr2 -= buffer_size
            
            #Annotates images
            """
            (ptA, ptB, ptC, ptD) = result.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(img, ptA, ptB, (0, 255, 0),60)
            cv2.line(img, ptB, ptC, (0, 255, 0), 60)
            cv2.line(img, ptC, ptD, (0, 255, 0), 60)
            cv2.line(img, ptD, ptA, (0, 255, 0), 60)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(result.center[0]), int(result.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = result.tag_family.decode("utf-8")
            cv2.putText(img, str(result.tag_id), (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] tag family: {}".format(str(result.hamming)))
            
            
            #Prints pose detection information
            print(result.pose_t*39.3701)
            """
        ct = monotonic()
        if ct-lastlook >= 1:
            lastlook = ct
            while cq[ptr1] < ct - 1 and ptr1 != ptr2:
                ptr1 += 1
                if(ptr1 >= buffer_size):
                    ptr1 -= buffer_size
            print('Average FPS:', (ptr2-ptr1)%buffer_size)
            # mt = monotonic()
            # t = mt-ct

            # print('Avg:', buffer_size/(mt-times[cnt%buffer_size]))
            # times[cnt%buffer_size] = mt

            # if(t != 0):
            #     print('Instant:',1/(t))
            # else:
            #     print('Subzero?')
        
        """
        # show the output image after AprilTag detection
        cv2.imshow("image", img)
        """
        
        cv2.imshow("image", img)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
        #Change exposure
        elif key in [ord('i'), ord('o')]:
            if key == ord('i'): exposure_time += increment
            if key == ord('o'): exposure_time -= increment
            print("Setting manual exposure, time:", exposure_time)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(exposure_time, 800)
            controlQueue.send(ctrl)