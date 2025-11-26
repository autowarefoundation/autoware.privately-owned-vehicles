#! /usr/bin/env python3
import pathlib
from argparse import ArgumentParser
import json
from PIL import Image, ImageDraw
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData


# Create coarse semantic segmentation mask
# of combined classes
def createMask(json_filepath, row, col):
    
    # Initializing colormap
    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    draw = ImageDraw.Draw(coarseSegColorMap) 

    # Colourmaps for classes
    sky_colour = (61, 184, 255)
    background_objects_colour = (61, 93, 255)
    vulnerable_living_colour = (255, 61, 61)
    small_mobile_vehicle_colour = (255, 190, 61)
    large_mobile_vehicle_colour = (255, 116, 61)
    road_edge_delimiter_colour = (216, 255, 61)
    road_colour = (0, 255, 220)

    
    # Open json file and get object labels and polygon points
    with open(json_filepath, 'r') as file: 
        annotations = json.load(file)
        
        for obj in annotations['objects']:
            
            label = obj['label']
            polygon = obj['polygon']
            coordinates = []

            for i in range(0, len(polygon)):
                    polygon[i][0] = int(polygon[i][0])
                    polygon[i][1] = int(polygon[i][1])
                    coordinates.append(int(polygon[i][0]))
                    coordinates.append(int(polygon[i][1]))

            # If there is a valid label
            if(len(polygon) > 1):
                # SKY
                if label == 'sky':
                    draw.polygon(coordinates, fill = sky_colour, outline = sky_colour) 

                # BACKGROUND OBJECTS
                elif label == 'billboard' or label == 'traffic sign' or label == 'traffic light' or \
                    label == 'pole' or label == 'obs-str-bar-fallback' or label == 'building' or \
                    label == 'bridge' or label == 'vegetation' or label == 'fallback background' or \
                    label == 'parking' or label == 'drivable-fallback' or label == 'sidewalk' or \
                    label == 'non-drivable fallback':
                    draw.polygon(coordinates, fill = background_objects_colour, outline = background_objects_colour) 

                # ROAD
                elif label == 'road':
                    draw.polygon(coordinates, fill = road_colour, outline = road_colour) 

                # ROAD EDGE DELIMITER
                elif label == 'curb' or label == 'wall' or label == 'fence' or label == 'guard rail':
                    draw.polygon(coordinates, fill = road_edge_delimiter_colour, outline = road_edge_delimiter_colour) 
                
                # VULNERABLE LIVING
                elif label == 'person' or label == 'animal':
                    draw.polygon(coordinates, fill = vulnerable_living_colour, outline = vulnerable_living_colour) 

                # SMALL MOBILE VEHICLE
                elif label == 'rider' or label == 'motorcycle' or label == 'bicycle':
                    draw.polygon(coordinates, fill = small_mobile_vehicle_colour, outline = small_mobile_vehicle_colour) 

                # LARGE MOBILE VEHICLE
                elif label == 'autorickshaw' or label == 'car' or label == 'truck' or label == 'bus' or \
                    label == 'caravan' or label == 'vehicle fallback':
                    draw.polygon(coordinates, fill = large_mobile_vehicle_colour, outline = large_mobile_vehicle_colour) 

    return coarseSegColorMap

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
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob(("*/*"))])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*/*")])

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

            # Open images and json files
            image = Image.open(str(images[index]))
            json_filepath = str(labels[index])

            # Get image size
            row, col = image.size

            # Create new Coarse Segmentation mask
            coarseSegColorMap = createMask(json_filepath, row, col)

            # Save images
            image.save(images_save_path + str(index) + ".png","PNG")
            coarseSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

            print(f'Processing image {index} of {num_images-1}')
    
        print('----- Processing complete -----')             

if __name__ == '__main__':
    main()