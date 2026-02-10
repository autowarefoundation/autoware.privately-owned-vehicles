#! /usr/bin/env python3

class CheckData():
    def __init__(self, num_images, num_gt):

        is_gt_path_valid = False
        is_image_path_valid = False
        is_data_valid = False
        self.check_passed = False

        # Checking if ground truth labels were read and logging error if missing
        if (num_gt > 0):
            print(f'Found {num_gt} ground truth samples')
            is_gt_path_valid = True
        else:
            raise ValueError('No ground truth samples found - check your ground truth data filepath:')

        # Checking if input images were read and logging error if missing
        if (num_images > 0):
            print(f'Found {num_images} input images')
            is_image_path_valid = True
        else:
            raise ValueError('No input images found - check your image data filepath')

        # Checking if number of ground truth labels matches number of input images
        if (num_images != num_gt):
            raise ValueError('Number of ground truth samples does not match number of input images:')
        else:
            is_data_valid = True
        
        # Final check
        if(is_gt_path_valid and is_image_path_valid and is_data_valid):
            self.check_passed = True

    def getCheck(self):
        return self.check_passed