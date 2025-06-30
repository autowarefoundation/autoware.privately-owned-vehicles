# The process_roadwork.py file 
The aim of the process_roadwork.py file is to create a folder of images and a corresponding folder of Binary labels. A label will have a value of 255 where the following original Class-Label values are present 
	cone
	drum	
	vertical_panel
	tubular_marker

and a value of 0 otherwise.

To perform the processing, Create a folder that would house the original unprocessed images and labels from the CMU RoadWork Website (https://kilthub.cmu.edu/articles/dataset/ROADWork_Data/26093197?file=47217583). Then all images from the images.zip folder of the CMU website should be stored in a folder named images in the created folder. Also, all labels from the zip folders sem_seg_labels.zip/gtFine/train and sem_seg_labels.zip/gtFine/val of the CMU website should be stored in a folder named gtFine.  

For example, if you created a folder called cmu_x, then it will have two sub folders, one called images which will have all the images and the other folder called gtFine which will have all the labels from the CMU Website.


To execute the code,
Assume the relative path to the cmu_x folder with the images/labels is "../../../../Data/Data_May_5th_2025/cmu_x/", this will be assigned to the variable -d in the python code.
The processed labels will be stored in a folder named "label", and the processed images will be stored in a folder named "image"; both folders (label and image) will be stored in a folder (relative path) name assigned to the variable -s. Assume we assign the variable -s as save;

Then execute the code below
python process_roadwork.py -d ../../../../Data/Data_May_5th_2025/cmu_x/  -s save/