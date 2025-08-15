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
    cnt = None
    if(len(contours > 0)):
      cnt = max(contours, key = lambda x: cv2.contourArea(x))
    return cnt

def make_visualization_freespace(prediction, image):

  colour_mask = np.array(image)

  # Getting freespace object labels
  free_space_labels = np.where(prediction == 2)

  # Get shape
  shape = prediction.shape
  row = shape[0]
  col = shape[1]

  # Create visualization
  binary_mask = np.zeros((row, col), dtype = "uint8")
  binary_mask[free_space_labels[0], free_space_labels[1]] = 255
  edge_contour = find_freespace_edge(binary_mask)
  if(edge_contour):
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
  parser.add_argument("-i", "--video_filepath", dest="video_filepath", help="path to input video which will be processed by SceneSeg")
  parser.add_argument("-o", "--output_file", dest="output_file", help="path to output video visualization file, must include output file name")
  parser.add_argument('-v', "--vis", action='store_true', default=False, help="flag for whether to show frame by frame visualization while processing is occuring")
  args = parser.parse_args() 

  # Saved model checkpoint path
  model_checkpoint_path = args.model_checkpoint_path
  model = SceneSegNetworkInfer(checkpoint_path=model_checkpoint_path)
  print('SceneSeg Model Loaded')
    
  # Create a VideoCapture object and read from input file
  # If the input is taken from the camera, pass 0 instead of the video file name.
  video_filepath = args.video_filepath
  cap = cv2.VideoCapture(video_filepath)

  # Output filepath
  output_filepath_obj = args.output_file + '.avi'
  output_filepath_freespace = args.output_file + 'freespace.avi'
  fps = cap.get(cv2.CAP_PROP_FPS)
  # Video writer object
  writer_obj = cv2.VideoWriter(output_filepath_obj,
    cv2.VideoWriter_fourcc(*"MJPG"), fps,(1280,720))
  
  writer_freespace = cv2.VideoWriter(output_filepath_freespace,
    cv2.VideoWriter_fourcc(*"MJPG"), fps,(1280,720))

  # Check if video catpure opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  else:
    print('Reading video frames')
  
  # Transparency factor
  alpha = 0.5
  
  # Read until video is completed
  print('Processing started')
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
 
      # Display the resulting frame
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image_pil = Image.fromarray(image)
      image_pil = image_pil.resize((640, 320))
      
      # Running inference
      prediction = model.inference(image_pil)
      vis_obj = make_visualization(prediction)
      vis_obj_freespace = make_visualization_freespace(prediction, image_pil)
      
      # Resizing to match the size of the output video
      # which is set to standard HD resolution
      frame = cv2.resize(frame, (1280, 720))
      vis_obj = cv2.resize(vis_obj, (1280, 720))
      vis_obj_freespace = cv2.resize(vis_obj_freespace, (1280, 720))

      # Create the composite visualization
      image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
      image_vis_freespace = cv2.addWeighted(vis_obj_freespace, alpha, frame, 1 - alpha, 0)

      if(args.vis):
        cv2.imshow('Prediction Objects', image_vis_obj)
        cv2.imshow('Prediction Freespace', image_vis_freespace)
        cv2.waitKey(10)

      # Writing to video frame
      writer_obj.write(image_vis_obj)
      writer_freespace.write(image_vis_freespace)

    else:
      print('Frame not read - ending processing')
      break

  # When everything done, release the video capture and writer objects
  cap.release()
  writer_obj.release()

  # Closes all the frames
  cv2.destroyAllWindows()
  print('Completed')

if __name__ == '__main__':
  main()
# %%