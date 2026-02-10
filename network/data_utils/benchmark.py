#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import time
import numpy as np
from argparse import ArgumentParser
from thop import profile
from pytorch_model_summary import summary
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.scene_3d_network import Scene3DNetwork
from model_components.ego_path_network import EgoPathNetwork

def benchmark(model, input_data, dtype='fp32', nwarmup=50, nruns=1000):

    if dtype=='fp16':
        input_data = input_data.half()
        model = model.half()

    print("Warm up ...")

    with torch.no_grad():
        for _ in range(nwarmup):
            _ = model(input_data)

    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []

    with torch.no_grad():
        for i in range(1, nruns+1):
            
            start_time = time.time()
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            
            timings.append(end_time - start_time)

            if i%100==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

def main():

    # Argument parser for data root path and save path
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="network_name", help="specify the name of the network which will be benchmarked")
    parser.add_argument("-p", "--precision", dest="precision",  default="fp32", help="set the precision of the network as fp32 or fp16")
    args = parser.parse_args()

    model_name = args.network_name
    precision = args.precision

    if(precision != 'fp16' and precision != 'fp32'):
        raise Exception("Please specify precision as fp32 or fp16")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # Instantiating Model and setting to evaluation mode
    model = 0
    
    if(model_name == 'SceneSeg'):
        model = SceneSegNetwork()
    elif (model_name == 'Scene3D'):
        sceneSegNetwork = SceneSegNetwork()
        model = Scene3DNetwork(sceneSegNetwork)
    elif (model_name == 'EgoPath'):
        sceneSegNetwork = SceneSegNetwork()
        model = EgoPathNetwork(sceneSegNetwork)
    else:
        raise Exception("Model name not specified correctly, please check")
    
    print(summary(model, torch.zeros((1, 3, 320, 640)), show_input=True))
    #model = model.to(device)
    #model = model.eval()

    # Fake input data
    #input_shape=(1, 3, 320, 640)
    #input_data = torch.randn(input_shape)
    #input_data = input_data.to(device)

    #total_ops, _ = profile(model, (input_data,), verbose=False)
    #print( model_name, "FLOPs(G): ", "%.2f" % (total_ops / (1000 ** 3)))

    # Run speed benchmark
    #benchmark(model, input_data, precision, 50, 1000)

if __name__ == '__main__':
  main()
# %%