#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.scene_seg_infer import SceneSegNetworkInfer

def find_freespace_edge(binary_mask):

    contours, _ = cv2.findContours(binary_mask,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    return cnt

def make_visualization_freespace(prediction, image):

  colour_mask = np.array(image)

  # Getting freespace object labels
  free_space_labels = np.where(prediction == 2)

  # Getting shape
  shape = prediction.shape
  row = shape[0]
  col = shape[1]

  # Creating visualization
  binary_mask = np.zeros((row, col), dtype = "uint8")
  binary_mask[free_space_labels[0], free_space_labels[1]] = 255
  if(edge_contour):
    edge_contour = find_freespace_edge(binary_mask)
    cv2.fillPoly(colour_mask, pts =[edge_contour], color=(28,255,145))

  # Converting to OpenCV BGR color channel ordering
  colour_mask = cv2.cvtColor(colour_mask, cv2.COLOR_RGB2BGR)

  return colour_mask

def make_visualization(prediction):

    # Creating visualization object
    shape = prediction.shape
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype = "uint8")

    # Assigning background colour
    vis_predict_object[:,:,0] = 255
    vis_predict_object[:,:,1] = 93
    vis_predict_object[:,:,2] = 61

    # Getting foreground object labels
    foreground_lables = np.where(prediction == 1)

    # Assigning foreground objects colour
    vis_predict_object[foreground_lables[0], foreground_lables[1], 0] = 145
    vis_predict_object[foreground_lables[0], foreground_lables[1], 1] = 28
    vis_predict_object[foreground_lables[0], foreground_lables[1], 2] = 255
            
    return vis_predict_object

def main(): 

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="path to input image which will be processed by SceneSeg")
    args = parser.parse_args() 

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = SceneSegNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('SceneSeg Model Loaded')
  
    # Transparency factor
    alpha = 0.5

    # Reading input image
    print('Reading Image')
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference and create visualization
    print('Running Inference and Creating Visualization')
    prediction = model.inference(image_pil)
    vis_obj = make_visualization(prediction)
    vis_obj_freespace = make_visualization_freespace(prediction, image_pil)

    # Resize and display visualization
    vis_obj = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
    vis_obj_freespace = cv2.resize(vis_obj_freespace, (frame.shape[1], frame.shape[0]))
    image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
    image_vis_freespace = cv2.addWeighted(vis_obj_freespace, alpha, frame, 1 - alpha, 0)
    cv2.imshow('Prediction Objects', image_vis_obj)
    cv2.imshow('Prediction Freespace', image_vis_obj)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
# %%