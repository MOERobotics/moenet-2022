import os
from multiprocessing import Queue
from enum import Enum
from utils.geom.geom3 import Rotation3D, Transform3D, Pose3D, Translation3D
import depthai as dai


class Camera:
    mxid: str
    type = Enum('type', {'default':None, 'S2':'S2', 'Lite':'Lite',})
    mode = Enum('mode', {'default':None, 'tag':'tag', 'object': 'object', 'hybrid': 'hybrid',})
    pose: Pose3D = Pose3D.zero()
    resolution = Enum('resolution', {'400p':400, '800p':800,})

    def tag_create_pipeline(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        xOutMono = pipeline.create(dai.node.XLinkOut)

        xOutMono.setStreamName("mono")

        if self.resolution == 400:
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        elif self.resolution == 800:
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        # Linking

        monoLeft.out.link(xOutMono.input)

        self.pipeline = pipeline

    def obj_create_pipeline(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        ctrl_in = pipeline.create(dai.node.XLinkIn)
        ctrl_out = pipeline.create(dai.node.XLinkOut)

        xoutNN.setStreamName("detections")
        ctrl_in.setStreamName("still_in")
        ctrl_out.setStreamName("still_out")


        # Properties
        camRgb.setPreviewSize(416, 416)
        camRgb.setStillSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        nnBlobPath = str((Path(__file__).parent / Path('./data/moeNetV1.blob')).resolve().absolute())
        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.75)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(3)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors([
            10.0,
            13.0,
            16.0,
            30.0,
            33.0,
            23.0,
            30.0,
            61.0,
            62.0,
            45.0,
            59.0,
            119.0,
            116.0,
            90.0,
            156.0,
            198.0,
            373.0,
            326.0])
        spatialDetectionNetwork.setAnchorMasks({
            "side52": [
                0,
                1,
                2
            ],
            "side26": [
                3,
                4,
                5
            ],
            "side13": [
                6,
                7,
                8
            ]
        })
        spatialDetectionNetwork.setIouThreshold(0.5)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        camRgb.still.link(ctrl_out.input)
        ctrl_in.out.link(camRgb.inputControl)

        spatialDetectionNetwork.out.link(xoutNN.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        

        self.pipeline = pipeline

    #NEED TO WRITE
    def hybrid_create_pipeline(self):
        pipeline = dai.Pipeline()
        self.pipeline = pipeline


def camera_init(info_file: str, apriltag_queue: Queue, object_queue: Queue):
    cam : Camera = Camera()
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
        if cam.type == 'S2':
            cam.resolution = 800
        else:
            cam.resolution = 400
    if cam.mode == Camera.mode.tag:
        cam.tag_create_pipeline()
    elif cam.mode == Camera.mode.object:
        cam.obj_create_pipeline()
    elif cam.mode == Camera.mode.hybrid:
        cam.hybrid_create_pipeline()
    
    cam._device = dai.Device(cam.create_pipeline(cam.pipeline))
    cam.device: dai.Device = cam._device.__enter__()

    if cam.mode == Camera.mode.tag:
        import apriltags
        cam.april_detector = apriltags.TagDetector(cam)
    elif cam.mode == Camera.mode.object:
        import objectdetect
        cam.object_detector = objectdetect.detector(cam)

    while True:
        if cam.mode == Camera.mode.tag:
            apriltag_queue.put(cam.april_detector.detect())
        elif cam.mode == Camera.mode.object:
            object_queue.put(cam.object_detector.detect())
    
    