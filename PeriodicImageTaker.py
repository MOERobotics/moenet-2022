#!/usr/bin/env python3

import cv2
import depthai as dai
import os
import time

#delay between captures
delay = 0.5

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(640,640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(60)

# Linking
camRgb.video.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # Set up folders to store images
    path = "./"
    path += input("Enter Folder Name Here: ")
    print("Creating ...")
    os.mkdir(path)
    print("Folder Created")

    count = 0
    ctime = time.time() - delay

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        frame = inRgb.getCvFrame()

        if time.time() - ctime >= delay:
            ctime = time.time()
            cv2.imwrite(path+'/'+str(count)+'.png', frame)
            print("Saved Image " + str(count))
            count += 1

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
