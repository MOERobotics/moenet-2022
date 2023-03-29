import depthai as dai

class detector():
    def init(self, cam):
        self.cam = cam
        self.xoutDetect = self.cam.device.getOutputQueue(name="detections", maxSize=1, blocking=False)
    def detect(self):
        label = self.cam.device.getQueueEvent(['detections', 'still_out'])
        detections: dai.SpatialImgDetections = self.xoutDetect.get()

        ntData = []
        for detection in detections.detections:
            label = detection.label
            x = detection.spatialCoordinates.x/1000
            y = detection.spatialCoordinates.y/1000
            z = detection.spatialCoordinates.z/1000

            ntData.extend([x,y,z,label])
        return ntData