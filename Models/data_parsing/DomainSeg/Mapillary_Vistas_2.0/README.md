# The process\_mapillary\_vistas.py file

The aim of the process\_mapillary\_vistas.py file is to create a folder of images and a corresponding folder of Binary labels. A label will have a value of 255 where the following original Class-Label values are present
object\_traffic\_cone
construction\_barrier\_temporary

and a value of 0 otherwise.



To perform processing, create a folder that would house the original unprocessed images and labels from the mapillary vistas Website (https://www.mapillary.com/dataset/vistas).

Assume the path to the folder with the original Mapillary Vistas training label is "../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/training/v2.0/labels/", this is assigned to -trlb in the python code

Assume the path to the folder with the original Mapillary Vistas training image is "../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/training/images/", this is assigned to -trim in the python code

Assume the path to the folder with the original Mapillary Vistas validation label is
"../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/validation/v2.0/labels/", this is assigned to -valb in the python code

Assume the path to the folder with the original Mapillary Vistas validation image is
"../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/validation/images/", this is assigned to -vaim in the python code

Assume the processed labels will be stored in a folder named "label", this is assigned to -lbs in the python code
Assume the processed images will be stored in a folder named "image", this is assigned to -ims in the python code

Assume the visualization (super imposed label on image) will be stored in a folder named "visualization", this is assigned to -vis in the python code





Then execute the code below



python process\_mapillary\_vistas.py -trlb ../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/training/v2.0/labels/ -trim ../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/training/images/  -valb ../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/validation/v2.0/labels/ -vaim ../../../../Data/Data\_May\_5th\_2025/Mapillary\_Vistas/validation/images/ -lbs label -ims image -vis visualization





A processed version of the data can be found in the Kaggle link https://www.kaggle.com/datasets/austinosas/mapillary-vistas-road-work-processed-data/data

