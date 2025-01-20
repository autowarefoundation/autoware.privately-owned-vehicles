#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import random
import sys
sys.path.append('..')
from data_utils.load_data_super_depth import LoadDataSuperDepth
from training.super_depth_trainer import SuperDepthTrainer


def main():

    # Root path
    root = '/mnt/media/SuperDepth/'

    # Model save path
    model_save_root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SuperDepth/2025_01_03/'

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

    # Pre-trained model checkpoint path
    root_path = \
        '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
    pretrained_checkpoint_path = root_path + 'iter_140215_epoch_4_step_15999.pth'

    # Trainer Class
    trainer = SuperDepthTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 50
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

        if(epoch == 3):
            batch_size = 16
        
        if(epoch == 5):
            batch_size = 8
        
        if(epoch == 8):
            batch_size = 5

        if(epoch >= 9 and epoch < 15):
            batch_size = 3

        if (epoch >= 15 and epoch < 25):
            batch_size = 2

        if (epoch >= 25):
            batch_size = 1

        # Loop through data
        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch

            # Reset iterators
            if(muad_count == muad_num_train_samples and \
                is_muad_complete == False):
                is_muad_complete =  True
                data_list.remove("MUAD")
            
            if(urbansyn_count == urbansyn_num_train_samples and \
                is_urbansyn_complete == False):
                is_urbansyn_complete = True
                data_list.remove("URBANSYN")

            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'MUAD' and \
                is_muad_complete == False):
                image, gt = muad_Dataset.getItemTrain(muad_count)
                muad_count += 1
            
            if(data_list[data_list_count] == 'URBANSYN' and \
               is_urbansyn_complete == False):
                image, gt = urbansyn_Dataset.getItemTrain(urbansyn_count)      
                urbansyn_count += 1

            # Assign Data
            trainer.set_data(image, gt)
            
            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

            # Run model and calculate loss
            trainer.run_model()
            
            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if((count+1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if((count+1) % 250 == 0):
                trainer.log_loss(log_count)
            
            # Logging Image to Tensor Board every 1000 steps
            if((count+1) % 1000 == 0):  
                trainer.save_visualization(log_count)
            
            # Save model and run validation on entire validation 
            # dataset after 8000 steps
            if((count+1) % 4000 == 0):

                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                
                trainer.save_model(model_save_path)

                # Validate
                print('Validating')

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                # Error
                running_mAE = 0

                # No gradient calculation
                with torch.no_grad():

                    # MUAD
                    for val_count in range(0, muad_num_val_samples):
                        image_val, gt_val = muad_Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val)

                        # Accumulating mAE score
                        running_mAE += mAE


                    # URBANSYN
                    for val_count in range(0, urbansyn_num_val_samples):
                        image_val, gt_val = urbansyn_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val)

                        # Accumulating mAE score
                        running_mAE += mAE

                    # LOGGING
                    # Calculating average loss of complete validation set
                    avg_mAE = running_mAE/total_val_samples
                        
                    # Logging average validation loss to TensorBoard
                    trainer.log_val_mAE(avg_mAE, log_count)

                # Resetting model back to training
                trainer.set_train_mode()

            data_list_count += 1

    trainer.cleanup()
 
    
if __name__ == '__main__':
    main()
# %%
