#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import random
from argparse import ArgumentParser
import sys
sys.path.append('..')
from data_utils.load_data_scene_seg import LoadDataSceneSeg
from training.scene_seg_trainer import SceneSegTrainer


def main():

    parser = ArgumentParser()
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path", help="root path where pytorch checkpoint file should be saved")
    parser.add_argument("-r", "--root", dest="root", help="root path to folder where data training data is stored")
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path =args.model_save_root_path

    # Data paths
    # ACDC
    acdc_labels_filepath= root + 'ACDC/gt_masks/'
    acdc_images_filepath = root + 'ACDC/images/'

    # IDDAW
    iddaw_labels_fileapath = root + 'IDDAW/gt_masks/'
    iddaw_images_fileapath = root + 'IDDAW/images/'

    # MUSES
    muses_labels_fileapath = root + 'MUSES/gt_masks/'
    muses_images_fileapath = root + 'MUSES/images/'

    # MAPILLARY
    mapillary_labels_fileapath = root + 'Mapillary_Vistas/gt_masks/'
    mapillary_images_fileapath = root + 'Mapillary_Vistas/images/'

    # COMMA10K
    comma10k_labels_fileapath = root + 'comma10k/gt_masks/'
    comma10k_images_fileapath = root + 'comma10k/images/'


    # ACDC - Data Loading
    acdc_Dataset = LoadDataSceneSeg(acdc_labels_filepath, acdc_images_filepath, 'ACDC')
    acdc_num_train_samples, acdc_num_val_samples = acdc_Dataset.getItemCount()

    # IDDAW - Data Loading
    iddaw_Dataset = LoadDataSceneSeg(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
    iddaw_num_train_samples, iddaw_num_val_samples = iddaw_Dataset.getItemCount()

    # MUSES - Data Loading
    muses_Dataset = LoadDataSceneSeg(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
    muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()

    # Mapillary - Data Loading
    mapillary_Dataset = LoadDataSceneSeg(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
    mapillary_num_train_samples, mapillary_num_val_samples = mapillary_Dataset.getItemCount()

    # comma10k - Data Loading
    comma10k_Dataset = LoadDataSceneSeg(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
    comma10k_num_train_samples, comma10k_num_val_samples = comma10k_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = acdc_num_train_samples + \
    + iddaw_num_train_samples + muses_num_train_samples \
    + mapillary_num_train_samples + comma10k_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = acdc_num_val_samples + \
    + iddaw_num_val_samples + muses_num_val_samples \
    + mapillary_num_val_samples + comma10k_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Trainer Class
    trainer = SceneSegTrainer()
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 10
    batch_size = 32

    # Epochs
    for epoch in range(0, num_epochs):

        # Iterators for datasets
        acdc_count = 0
        iddaw_count = 0
        muses_count = 0
        comma10k_count = 0
        mapillary_count = 0

        is_acdc_complete = False
        is_iddaw_complete = False
        is_muses_complete = False
        is_mapillary_complete = False
        is_comma10k_complete = False

        data_list = []
        data_list.append('ACDC')
        data_list.append('IDDAW')
        data_list.append('MUSES')
        data_list.append('MAPILLARY')
        data_list.append('COMMA10K')
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


        # Loop through data
        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch

            # Reset iterators
            if(acdc_count == acdc_num_train_samples and \
               is_acdc_complete == False):
                is_acdc_complete =  True
                data_list.remove("ACDC")
            
            if(iddaw_count == iddaw_num_train_samples and \
               is_iddaw_complete == False):
                is_iddaw_complete = True
                data_list.remove("IDDAW")
            
            if(muses_count == muses_num_train_samples and \
                is_muses_complete == False):
                is_muses_complete = True
                data_list.remove('MUSES')
            
            if(mapillary_count == mapillary_num_train_samples and \
               is_mapillary_complete == False):
                is_mapillary_complete = True
                data_list.remove('MAPILLARY')

            if(comma10k_count == comma10k_num_train_samples and \
               is_comma10k_complete == False):
                is_comma10k_complete = True
                data_list.remove('COMMA10K')

            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'ACDC' and \
                is_acdc_complete == False):
                image, gt, class_weights = \
                        acdc_Dataset.getItemTrain(acdc_count)
                acdc_count += 1
            
            if(data_list[data_list_count] == 'IDDAW' and \
               is_iddaw_complete == False):
                image, gt, class_weights = \
                    iddaw_Dataset.getItemTrain(iddaw_count)      
                iddaw_count += 1

            if(data_list[data_list_count] == 'MUSES' and \
               is_muses_complete == False):
                image, gt, class_weights = \
                    muses_Dataset.getItemTrain(muses_count)
                muses_count += 1
            
            if(data_list[data_list_count] == 'MAPILLARY' and \
               is_mapillary_complete == False):
                image, gt, class_weights = \
                    mapillary_Dataset.getItemTrain(mapillary_count)
                mapillary_count +=1
            
            if(data_list[data_list_count] == 'COMMA10K' and \
                is_comma10k_complete == False):
                image, gt, class_weights = \
                    comma10k_Dataset.getItemTrain(comma10k_count)
                comma10k_count += 1
            
            # Assign Data
            trainer.set_data(image, gt, class_weights)
            
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
            if((count+1) % 8000 == 0):
                
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

                running_IoU_full = 0
                running_IoU_bg = 0
                running_IoU_fg = 0
                running_IoU_rd = 0

                # No gradient calculation
                with torch.no_grad():

                    # ACDC
                    for val_count in range(0, acdc_num_val_samples):
                        image_val, gt_val, _ = \
                            acdc_Dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # MUSES
                    for val_count in range(0, muses_num_val_samples):
                        image_val, gt_val, _ = \
                            muses_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    
                    # IDDAW
                    for val_count in range(0, iddaw_num_val_samples):
                        image_val, gt_val, _ = \
                            iddaw_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # MAPILLARY
                    for val_count in range(0, mapillary_num_val_samples):
                        image_val, gt_val, _ = \
                            mapillary_Dataset.getItemVal(val_count)
                        
                         # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # COMMA10K
                    for val_count in range(0, comma10k_num_val_samples):
                        image_val, gt_val, _ = \
                            comma10k_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    
                    # Calculating average loss of complete validation set
                    mIoU_full = running_IoU_full/total_val_samples
                    mIoU_bg = running_IoU_bg/total_val_samples
                    mIoU_fg = running_IoU_fg/total_val_samples
                    mIoU_rd = running_IoU_rd/total_val_samples
                    
                    # Logging average validation loss to TensorBoard
                    trainer.log_IoU(mIoU_full, mIoU_bg, mIoU_fg, mIoU_rd, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
                
            data_list_count += 1

    trainer.cleanup()


if __name__ == '__main__':
    main()
# %%
