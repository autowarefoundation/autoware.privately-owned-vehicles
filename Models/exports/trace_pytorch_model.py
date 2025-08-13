#!/usr/bin/python3
import torch
from argparse import ArgumentParser
import sys
sys.path.append('../..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.scene_3d_network import Scene3DNetwork
from model_components.domain_seg_network import DomainSegNetwork
from model_components.ego_path_network import EgoPathNetwork


##
## Example Usage: "python3 traced_script_module_save.py -n SceneSeg -p _checkpoint_file_.pth  -o _output_trace_file.pt"
##

def main(): 

    # Command line arguments
    parser = ArgumentParser()
        
    parser.add_argument("-n", "--name", dest="network_name", required=True, \
                        help="specify the name of the network which will be benchmarked")

    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True, \
                        help="path to pytorch checkpoint file to load model dict")
    
    parser.add_argument("-o", "--output_pt_trace_filepath", dest="output_pt_trace_filepath", required=True, \
                        help="path to *.pt output trace file generated")
    
    args = parser.parse_args() 

    # Model name, saved model checkpoint path and traced model save path
    model_name = args.network_name
    model_checkpoint_path = args.model_checkpoint_path
    traced_model_save_path = args.output_pt_trace_filepath

    # Checking devices (GPU vs CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO: Using {device} for inference.')
        
    # Instantiating Model and setting to evaluation mode
    model = 0
    if(model_name == 'SceneSeg'):
        print('Processing SceneSeg Network')
        model = SceneSegNetwork()
    elif (model_name == 'Scene3D'):
        print('Processing Scene3D Network')
        sceneSegNetwork = SceneSegNetwork()
        model = Scene3DNetwork(sceneSegNetwork)
    elif (model_name == 'EgoPath'):
        print('Processing EgoPath Network')
        sceneSegNetwork = SceneSegNetwork()
        model = EgoPathNetwork(sceneSegNetwork)
    elif (model_name == 'DomainSeg'):
        print('Processing DomainSeg Network')
        sceneSegNetwork = SceneSegNetwork()
        model = DomainSegNetwork(sceneSegNetwork)
    else:
        raise Exception("Model name not specified correctly, please check")
    
    # Loading Pytorch checkpoint
    print('Loading Network')
    if(len(model_checkpoint_path) > 0):
            model.load_state_dict(torch.load \
                (model_checkpoint_path, weights_only=True, map_location=device))
    else:
        raise ValueError('No path to checkpiont file provided in class initialization')
    model = model.to(device)
    model = model.eval()

    # Fake input data
    input_shape=(1, 3, 320, 640)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)

    # Torch Export
    # Run and Trace the model with input image
    print('Tracing model')
    traced_script_module = torch.jit.trace(model, input_data)
    traced_script_module.save(traced_model_save_path) 
    print("Torch Trace Export file generated successfully.")


if __name__ == '__main__':
    main()