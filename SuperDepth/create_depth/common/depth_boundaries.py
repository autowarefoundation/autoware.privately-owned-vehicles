#! /usr/bin/env python3
import numpy as np

class DepthBoundaries():
    def __init__(self, depth_map, threshold = 8):
        
        # Getting size of depth map
        size = depth_map.shape
        height = size[0]
        width = size[1]

        # Initializing depth boundary mask
        self.depth_boundaries = np.zeros_like(depth_map, dtype=np.uint8)

        # Fiding depth boundaries
        for i in range(1, height-1):
            for j in range(1, width-1):

                # Finding derivative
                x_grad = depth_map[i-1,j] - depth_map[i+1, j]
                y_grad = depth_map[i,j-1] - depth_map[i, j+1]
                grad = abs(x_grad) + abs(y_grad)
                
                # Derivative threshold
                if(grad > threshold and depth_map[i-1, j] != 0):
                    self.depth_boundaries[i,j] = 255

    def getDepthBoundaries(self):
        return self.depth_boundaries
   