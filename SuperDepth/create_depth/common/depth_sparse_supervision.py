#! /usr/bin/env python3
import numpy as np

class DepthSparseSupervision():
    def __init__(self, image, height_map, max_height, min_height, threshold = 25):
            
        # Getting size of height map
        size = height_map.shape
        height = size[0]
        width = size[1]

        # Initializing depth boundary mask
        self.sparse_supervision = np.zeros_like(height_map)

        # Getting pixel access for image
        px = image.load()

        # Fiding image gradients
        for i in range(1, width-1):
            for j in range(1, height-1):

                # Finding image derivative - using green colour channel
                x_grad = px[i-1,j][1] - px[i+1, j][1]
                y_grad = px[i,j-1][1] - px[i, j+1][1]
                grad = abs(x_grad) + abs(y_grad)

                # Derivative threshold to find likely stereo candidates
                if(grad > threshold):
                    self.sparse_supervision[j,i] = height_map[j,i]
                else:
                    self.sparse_supervision[j,i] = max_height

                # Max height threshold
                if(height_map[j,i] == max_height):
                    self.sparse_supervision[j,i] = max_height
        
        # Clipping height values for dataset
        self.sparse_supervision = self.sparse_supervision.clip(min = min_height, max = max_height)

    def getSparseSupervision(self):
        return self.sparse_supervision
   