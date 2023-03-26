from multiprocessing import set_start_method, Process, Queue
from queue import Empty
from time import sleep
import fnmatch
import os
from camera import camera_init

#How often messages are sent to Network Tables in Hz
update_frequency = 100

if __name__ == '__main__':
    #Important - Default on Linux is fork which doesn't work well with depthai
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

    from apriltags import combine_detections, network_format
    import Network_Tables_Sender as nts
    
    while True:
        # Collect data from child processes and send to Network tables
        #Detections have the following format - list({detection})
        apriltags = []
        for q in apriltag_queue:
            current_apriltags : list(apriltags.detections)
            while True:
                try:
                    item = q.get(block=False)
                except Empty:
                    break
                current_apriltags.append(item)
            apriltags.extend(current_apriltags)
        
        if len(apriltags) != 0:
            #Combine poses
            pose = combine_detections(apriltags)
            nts.send_pose(network_format(pose))

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

        # Pause for given time
        sleep(1/update_frequency)