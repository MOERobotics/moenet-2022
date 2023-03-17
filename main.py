import json
import Network_Tables_Sender as nts
from multiprocessing import set_start_method, Process, Queue
from queue import Empty
from time import sleep, monotonic
import fnmatch
import os
from camera import camera_init
from apriltags import combine

if __name__ == '__main__':
    set_start_method('spawn')
    camera_data = fnmatch.filter(os.listdir('./data/'), 'Cam_*.json')
    
    apriltag_queue = []
    object_queue = []
    
    for cam in camera_data:
        apriltag_queue.append(Queue())
        object_queue.append(Queue())
        io_proc = Process(
                    target=camera_init,
                    args=(cam, apriltag_queue[-1], object_queue[-1]),
                    daemon=True
                )
        io_proc.start()
    while True:
        # Collect data from child processes and send to Network tables
        #Detections have the following format - {translation, rotation, error}
        #Will send None object prior to each detection cycle
        apriltags = []
        for q in apriltag_queue:
            current_apriltags = []
            while True:
                try:
                    item = q.get(block=False)
                except Empty:
                    break
                if item is None:
                    current_apriltags = []
                else:
                    current_apriltags.append(item)
            apriltags.extend(current_apriltags)
        
        if len(apriltags) != 0:
            #Combine poses
            pose = combine(apriltags)
            nts.send_pose(pose)

        objects = []
        for q in object_queue:
            current_objects = []
            while True:
                try:
                    current_objects = q.get(block = False)
                except Empty:
                    break
            objects.extend(current_objects)

        if len(objects) != 0:
            nts.send_detections(objects)

        # Pause for 10ms
        sleep(0.01)