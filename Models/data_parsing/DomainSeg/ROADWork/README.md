## process_roadwork.py

The aim of the **process_roadwork.py** script is to create a folder of images and a corresponding folder of binary segmentation labels. A label will have a value of 255 where the following original Class-Label values are present:

- cone
- drum	
- vertical_panel
- tubular_marker

and a value of 0 otherwise.

### Dataset preparation

To perform the processing, download the [CMU ROADWork dataset](https://kilthub.cmu.edu/articles/dataset/ROADWork_Data/26093197?file=47217583) and unzip the downloaded data into a directory. 

- All images from the **images.zip** folder of the ROADWork dataset should be stored in a folder named **images** within the directory you created. 
- All labels from the zip folders **sem_seg_labels.zip/gtFine/train** and **sem_seg_labels.zip/gtFine/val** of the CMU ROADWork dataset should be stored in a folder named **gtFine** within the same directory you created.  

For example, if you created a folder called cmu_x, then it will have two sub folders, one called images which will have all the images and the other folder called gtFine which will have all the labels from the ROADWork Dataset:

### Example directory structure

**cmu_x**
- images (containing all images)
- labels (containing all labels)


To execute the code, asume the relative path to the cmu_x folder with the images/labels is "../../../../Data/Data_May_5th_2025/cmu_x/", this will be assigned to the variable -d in the python code.

The processed labels will be stored in a folder named "label", and the processed images will be stored in a folder named "image"; both folders (label and image) will be stored in a folder (relative path) name assigned to the variable -s. Assume we assign the variable -s as save;

### Example Usage:

```bash
python3 process_roadwork.py -d ../../../../Data/Data_May_5th_2025/cmu_x/  -s save/
```

A processed version of the data can be found in the Kaggle link https://www.kaggle.com/datasets/austinosas/cmu-road-work-processed-data

