#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.domain_seg_infer import DomainSegNetworkInfer


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
  foreground_lables = np.where(prediction == 1.0)

  # Assigning foreground objects colour
  vis_predict_object[foreground_lables[0], foreground_lables[1], 0] = 28
  vis_predict_object[foreground_lables[0], foreground_lables[1], 1] = 148
  vis_predict_object[foreground_lables[0], foreground_lables[1], 2] = 255
        
  return vis_predict_object

def main(): 

  parser = ArgumentParser()
  parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
  parser.add_argument("-i", "--video_filepath", dest="video_filepath", help="path to input video which will be processed by DomainSeg")
  parser.add_argument("-o", "--output_file", dest="output_file", help="path to output video visualization file, must include output file name")
  parser.add_argument('-v', "--vis", action='store_true', default=False, help="flag for whether to show frame by frame visualization while processing is occuring")
  args = parser.parse_args() 

  # Saved model checkpoint path
  model_checkpoint_path = args.model_checkpoint_path
  model = DomainSegNetworkInfer(checkpoint_path=model_checkpoint_path)
  print('DomainSeg Model Loaded')
    
  # Create a VideoCapture object and read from input file
  # If the input is taken from the camera, pass 0 instead of the video file name.
  video_filepath = args.video_filepath
  cap = cv2.VideoCapture(video_filepath)

  # Output filepath
  output_filepath_obj = args.output_file + '.avi'
  fps = cap.get(cv2.CAP_PROP_FPS)
  # Video writer object
  writer_obj = cv2.VideoWriter(output_filepath_obj,
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
      
      # Resizing to match the size of the output video
      # which is set to standard HD resolution
      frame = cv2.resize(frame, (1280, 720))
      vis_obj = cv2.resize(vis_obj, (1280, 720))

      # Create the composite visualization
      image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)

      if(args.vis):
        cv2.imshow('Prediction Objects', image_vis_obj)
        cv2.waitKey(10)

      # Writing to video frame
      writer_obj.write(image_vis_obj)

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