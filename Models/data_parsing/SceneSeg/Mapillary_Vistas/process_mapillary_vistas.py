#! /usr/bin/env python3
import pathlib
from argparse import ArgumentParser
from PIL import Image
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData

# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px = colorMap.load()
    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Colourmaps for classes
    sky_colour = (61, 184, 255)
    background_objects_colour = (61, 93, 255)
    vulnerable_living_colour = (255, 61, 61)
    small_mobile_vehicle_colour = (255, 190, 61)
    large_mobile_vehicle_colour = (255, 116, 61)
    road_edge_delimiter_colour = (216, 255, 61)
    road_colour = (0, 255, 220)

    is_valid_image = True
    road_sum = 0
    drivable_other_sum = 0
    
    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):
                       
            # SKY
            if px[x,y] == 27:
                cx[x,y] = sky_colour

            # BACKGROUND OBJECTS
            # Building    
            elif px[x,y] == 17:
                cx[x,y] = background_objects_colour
            # Pole  
            elif px[x,y] == 45 or px[x,y] == 47:
                cx[x,y] = background_objects_colour
            # Traffic Light
            elif px[x,y] == 48:
                cx[x,y] = background_objects_colour 
            # Traffic Sign
            elif px[x,y] == 50:
                cx[x,y] = background_objects_colour 
            # Traffic Sign Back
            elif px[x,y] == 49:
                cx[x,y] = background_objects_colour 
            # Traffic Sign Frame
            elif px[x,y] == 46:
                cx[x,y] = background_objects_colour
            # Vegetation
            elif px[x,y] == 30:
                cx[x,y] = background_objects_colour
            # Terrain
            elif px[x,y] == 29:
                cx[x,y] = background_objects_colour
            # Bird
            elif px[x,y] == 0:
                cx[x,y] = background_objects_colour
            # Parking
            elif px[x,y] == 10:
                cx[x,y] = background_objects_colour
                # Checking Validity of Image for on-road scene
                drivable_other_sum += 1
            # Pedestrian Area
            elif px[x,y] == 11:
                cx[x,y] = background_objects_colour
                # Checking Validity of Image for on-road scene
                drivable_other_sum += 1
            # Rail Track
            elif px[x,y] == 12:
                cx[x,y] = background_objects_colour
                # Checking Validity of Image for on-road scene
                drivable_other_sum += 1
            # Sidewalk
            elif px[x,y] == 15:
                cx[x,y] = background_objects_colour
                # Checking Validity of Image for on-road scene
                drivable_other_sum += 1
            # Bridge
            elif px[x,y] == 16:
                cx[x,y] = background_objects_colour
            # Tunnel
            elif px[x,y] == 18:
                cx[x,y] = background_objects_colour   
            # Mountain
            elif px[x,y] == 25:
                cx[x,y] = background_objects_colour
            # Sand
            elif px[x,y] == 26:
               cx[x,y] = background_objects_colour
            # Snow
            elif px[x,y] == 28:
                cx[x,y] = background_objects_colour
                # Snow images in this dataset conflict with
                # snow images in other datasets especially
                # for snowy road surfaces
                is_valid_image = False
            # Water
            elif px[x,y] == 31:
                cx[x,y] = background_objects_colour
            # Banner
            elif px[x,y] == 32:
                cx[x,y] = background_objects_colour
            # Bench
            elif px[x,y] == 33:
                cx[x,y] = background_objects_colour
            # Bike Rack
            elif px[x,y] == 34:
                cx[x,y] = background_objects_colour
            # Billboard
            elif px[x,y] == 35:
                cx[x,y] = background_objects_colour
            # CCTV Camera
            elif px[x,y] == 37:
                cx[x,y] = background_objects_colour 
            # Fire Hydrant
            elif px[x,y] == 38:
                cx[x,y] = background_objects_colour 
            # Junction Box
            elif px[x,y] == 39:
                cx[x,y] = background_objects_colour 
            # Mail Box
            elif px[x,y] == 40:
                cx[x,y] = background_objects_colour 
            # Phone Booth
            elif px[x,y] == 42:
                cx[x,y] = background_objects_colour  
            # Pothole
            elif px[x,y] == 43:
                cx[x,y] = background_objects_colour
            # Street Light
            elif px[x,y] == 44:
                cx[x,y] = background_objects_colour
            # Trash Can
            elif px[x,y] == 51:
                cx[x,y] = background_objects_colour
            # Ego Vehicle
            elif px[x,y] == 63 or px[x,y] == 64:
                cx[x,y] = background_objects_colour

            # VULNERABLE LIVING
            # Person
            elif px[x,y] == 19:
                cx[x,y] = vulnerable_living_colour
            # Animal
            elif px[x,y] == 1:
                cx[x,y] = vulnerable_living_colour    

            # SMALL MOBILE VEHICLE
            # Rider
            elif px[x,y] == 20 or px[x,y] == 21 \
                or px[x,y] == 22:
                cx[x,y] = small_mobile_vehicle_colour
            # Motorcylce
            elif px[x,y] == 57:
                cx[x,y] = small_mobile_vehicle_colour
            # Bicycle
            elif px[x,y] == 52:
                cx[x,y] = small_mobile_vehicle_colour

            # LARGE MOBILE VEHICLE
            # Car
            elif px[x,y] == 55:
                cx[x,y] = large_mobile_vehicle_colour
            # Truck
            elif px[x,y] == 61:
                cx[x,y] = large_mobile_vehicle_colour
            # Bus
            elif px[x,y] == 54:
                cx[x,y] = large_mobile_vehicle_colour
            # Train
            elif px[x,y] == 58:
                cx[x,y] = large_mobile_vehicle_colour
            # Boat
            elif px[x,y] == 53:
                cx[x,y] = large_mobile_vehicle_colour
            # Caravan
            elif px[x,y] == 56:
                cx[x,y] = large_mobile_vehicle_colour
            # Other Vehicle
            elif px[x,y] == 59:
                cx[x,y] = large_mobile_vehicle_colour
            # Trailer
            elif px[x,y] == 60:
                cx[x,y] = large_mobile_vehicle_colour
            # Wheeled slow
            elif px[x,y] == 62:
               cx[x,y] = large_mobile_vehicle_colour 
            
            # ROAD EDGE DELIMITER
            # Curb
            elif px[x,y] == 2:
                cx[x,y] = road_edge_delimiter_colour
            # Wall
            elif px[x,y] == 6:
                cx[x,y] = road_edge_delimiter_colour
            # Fence
            elif px[x,y] == 3:
                cx[x,y] = road_edge_delimiter_colour
            # Guard Rail
            elif px[x,y] == 4:
                cx[x,y] = road_edge_delimiter_colour
            # Barrier
            elif px[x,y] == 5:
                cx[x,y] = road_edge_delimiter_colour
            # Curb Cut
            elif px[x,y] == 9:
                cx[x,y] = road_edge_delimiter_colour

            # ROAD
            elif px[x,y] == 13:
                cx[x,y] = road_colour
                road_sum+=1
            # Bike Lane
            elif px[x,y] == 7:
                cx[x,y] = road_colour   
            # Cross Walk 
            elif px[x,y] == 8:
                cx[x,y] = road_colour     
            # Service Lane
            elif px[x,y] == 14:
                cx[x,y] = road_colour   
            # Cross Walk Lane Marking
            elif px[x,y] == 23:
                cx[x,y] = road_colour   
            # General Lane Marking 
            elif px[x,y] == 24:
                cx[x,y] = road_colour 
            # Catch Basin  
            elif px[x,y] == 36:
               cx[x,y] = road_colour 
            # Man Hole
            elif px[x,y] == 41:
               cx[x,y] = road_colour  

    if (road_sum <= drivable_other_sum):
        is_valid_image = False

    return coarseSegColorMap, is_valid_image

def main():

    parser = ArgumentParser()
    parser.add_argument("-l", "--labels", dest="labels_filepath", help="path to folder with input ground truth labels")
    parser.add_argument("-i", "--images", dest="images_filepath", help="path to folder with input images")
    parser.add_argument("-ls", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-is", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()

    # Paths to read input images and ground truth label masks from training data
    labels_filepath = args.labels_filepath
    images_filepath = args.images_filepath

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = args.labels_save_path
    images_save_path = args.images_save_path
    
    # Reading dataset labels and images and sorting returned list in alphabetical order
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.jpg")])

    # Getting number of labels and images
    num_labels = len(labels)
    num_images = len(images)

    # Check if sample numbers are correct
    check_data = CheckData(num_images, num_labels)
    check_passed = check_data.getCheck()
    
    # If all data checks have been passed
    if(check_passed):

        print('Beginning processing of data')

        # Looping through data
        for index in range(0, num_images):
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            label = Image.open(str(labels[index]))

            row, col = image.size
            half_res = (int(row/2), int(col/2))
            image = image.resize(half_res)
            label = label.resize(half_res)

            # Create new Coarse Segmentation mask
            coarseSegColorMap, is_valid_image = createMask(label)

            # Save images
            if(is_valid_image):
                image.save(images_save_path + str(index) + ".png","PNG")
                coarseSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

            print(f'Processing image {index} of {num_images-1}')

        print('----- Processing complete -----')

if __name__ == '__main__':
    main()