#! /usr/bin/env python3
import cv2
import numpy as np

class LidarDepthFill():
    def __init__(self, depth_map):

        self.depth_map = depth_map

        # Filter kernels
        erosion_kernel = np.ones((3,3), np.uint8) 
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))

        # Morphology filters
        self.depth_map = cv2.dilate(self.depth_map, erosion_kernel, iterations=3) 
        self.depth_map = cv2.morphologyEx(self.depth_map, cv2.MORPH_CLOSE, closing_kernel)
        # Median blur
        self.depth_map = cv2.medianBlur(self.depth_map, 5)

    # Get filled in and interpolated lidar depth map
    def getDepthMap(self):
        return self.depth_map
    
