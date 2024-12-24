#! /usr/bin/env python3
import numpy as np

class HeightMap():
    def __init__(self, depth_map, max_height, min_height, 
                 camera_height, focal_length, cy):
        
        # Getting size of depth map
        size = depth_map.shape
        height = size[0]
        width = size[1]

        self.minimumHeight = 0

        # Initializing height-map
        self.height_map = np.zeros_like(depth_map)

        for i in range(0, height):
            for j in range(0, width):
                depth_val = depth_map[i, j]
                H = (cy-i)*(depth_val)/focal_length
                self.height_map[i,j] = H + camera_height
        
        # Get minimum height before clipping
        self.minimumHeight = np.min(self.height_map)

        # Clipping height values for dataset
        self.height_map = self.height_map.clip(min = min_height, max = max_height)

    def getHeightMap(self):
        return self.height_map
    
    def getMinimumHeight(self):
        return self.minimumHeight