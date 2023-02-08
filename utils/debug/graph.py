from .base import Debugger, DebugFrame, ReferenceFrame, ItemId
import matplotlib.pyplot as plt
import numpy as np

class GraphDebug(Debugger):
    def __init__(self, rf: ReferenceFrame, item: ItemId, buffer_len: int = 100) -> None:
        super().__init__()

        self.buffer = list()
        self.buffer_len = buffer_len
        self.rf = rf
        self.item = item
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)

    def finish_frame(self, frame: DebugFrame):
        if len(self.buffer) > self.buffer_len:
            self.buffer = self.buffer[-20:]
        
        pose = frame[self.item, self.rf]
        if pose is None:
            return
        
        self.buffer.append(pose[0])

        ax1 = self.ax1
        b = np.array(self.buffer)
        ax1.cla()
        ax1.set(xlim=(-5,5), ylim=(-5,5))
        ax1.scatter([0], [0])
        ax1.plot(b[:,0], b[:,1])
        plt.draw()
        plt.pause(.001)