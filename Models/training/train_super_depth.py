#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import random
from pytorch_model_summary import summary
from PIL import Image
import sys
sys.path.append('..')
from model_components.super_depth_network import SuperDepthNetwork
from model_components.scene_seg_network import SceneSegNetwork
from data_utils.load_data_super_depth import LoadDataSuperDepth
from training.super_depth_trainer import SuperDepthTrainer


def main():

    # Root path
    root = '/mnt/media/SuperDepth/'

    # Model save path
    model_save_root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SuperDepth/'

    # Data paths
    # MUAD
    muad_labels_filepath= root + 'MUAD/height/'
    muad_images_filepath = root + 'MUAD/image/'

    # URBANSYN
    urbansyn_labels_fileapath = root + 'UrbanSyn/height/'
    urbansyn_images_fileapath = root + 'UrbanSyn/image/'

    # MUAD - Data Loading
    muad_Dataset = LoadDataSuperDepth(muad_labels_filepath, muad_images_filepath, 'MUAD')
    muad_num_train_samples, muad_num_val_samples = muad_Dataset.getItemCount()

    # URBANSYN - Data Loading
    urbansyn_Dataset = LoadDataSuperDepth(urbansyn_labels_fileapath, urbansyn_images_fileapath, 'URBANSYN')
    urbansyn_num_train_samples, urbansyn_num_val_samples = urbansyn_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = muad_num_train_samples + \
    + urbansyn_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = muad_num_val_samples + \
    + urbansyn_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Trainer Class
    trainer = SuperDepthTrainer()
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 10
    batch_size = 32

    # Epochs
    for epoch in range(0, num_epochs):

        # Iterators for datasets
        muad_count = 0
        urbansyn_count = 0

        is_muad_complete = False
        is_urbansyn_complete = False
        
        data_list = []
        data_list.append('MUAD')
        data_list.append('URBANSYN')
        random.shuffle(data_list)
        data_list_count = 0

        if(epoch == 1):
            batch_size = 16
        
        if(epoch == 2):
            batch_size = 8
        
        if(epoch == 3):
            batch_size = 5

        if(epoch >= 4 and epoch < 6):
            batch_size = 3

        if (epoch >= 6 and epoch < 8):
            batch_size = 2

        if (epoch > 8):
            batch_size = 1


    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # Load pre-trained weights
    sceneSegNetwork = SceneSegNetwork()
    root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
    pretrained_checkpoint_path = root_path + 'iter_140215_epoch_4_step_15999.pth'
    sceneSegNetwork.load_state_dict(torch.load \
        (pretrained_checkpoint_path, weights_only=True, map_location=device))
    
    # Instantiate Model with pre-trained weights
    model = SuperDepthNetwork(sceneSegNetwork)
    print(summary(model, torch.zeros((1, 3, 320, 640)), show_input=True))
    model = model.to(device)

    # Random input
    input_image_filepath = '/mnt/media/SuperDepth/UrbanSyn/image/10.png'
    image = Image.open(input_image_filepath)
    image = image.resize((640, 320))

    # Image loader
    image_loader = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]
    )

    image_tensor = image_loader(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    prediction = model(image_tensor)

    prediction = prediction.squeeze(0).cpu().detach()
    prediction = prediction.permute(1, 2, 0)

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(prediction)
    
    
if __name__ == '__main__':
    main()
# %%
