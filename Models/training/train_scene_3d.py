#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import random
import sys
sys.path.append('..')
from data_utils.load_data_scene_3d import LoadDataScene3D
from training.scene_3d_trainer import Scene3DTrainer

def main():

    # Root path
    root = '/mnt/media/Scene3D/'

    # Model save path
    model_save_root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/2025_02_16/model/'

    # Data paths

    # KITTI
    kitti_labels_filepath = root + 'KITTI/depth/'
    kitti_images_filepath = root + 'KITTI/image/'
    kitti_validities_filepath = root + 'KITTI/validity/'
    s_kitti = 1.556

    # DDAD
    ddad_labels_filepath = root + 'DDAD/depth/'
    ddad_images_filepath = root + 'DDAD/image/'
    ddad_validities_filepath = root + 'DDAD/validity/'
    s_ddad_f = 0.805
    s_ddad_b = 1.452

    # URBANSYN
    urbansyn_labels_fileapath = root + 'UrbanSyn/depth/'
    urbansyn_images_fileapath = root + 'UrbanSyn/image/'
    s_urbansyn = 1.068

    # MUAD
    muad_labels_fileapath = root + 'MUAD/Audited/depth/'
    muad_images_fileapath = root + 'MUAD/Audited/image/'
    s_muad = 1.068

    # GTAV
    gta_labels_fileapath = root + 'GTAV/depth/'
    gta_images_fileapath = root + 'GTAV/image/'
    s_gta = 1.5708

    # KITTI - Data Loading
    kitti_Dataset = LoadDataScene3D(kitti_labels_filepath, kitti_images_filepath, 
                                           'KITTI', kitti_validities_filepath)
    kitti_num_train_samples, kitti_num_val_samples = kitti_Dataset.getItemCount()

    # DDAD - Data Loading
    ddad_Dataset = LoadDataScene3D(ddad_labels_filepath, ddad_images_filepath, 
                                           'DDAD', ddad_validities_filepath)
    ddad_num_train_samples, ddad_num_val_samples = ddad_Dataset.getItemCount()
    ddad_train_cams, ddad_val_cams = ddad_Dataset.getDDADCameras()
   
    # URBANSYN - Data Loading
    urbansyn_Dataset = LoadDataScene3D(urbansyn_labels_fileapath, urbansyn_images_fileapath, 'URBANSYN')
    urbansyn_samples = urbansyn_Dataset.getTotalCount()

    # URBANSYN - Data Loading
    muad_Dataset = LoadDataScene3D(muad_labels_fileapath, muad_images_fileapath, 'MUAD')
    muad_samples = muad_Dataset.getTotalCount()

    # GTAV - Data Loading
    gta_Dataset = LoadDataScene3D(gta_labels_fileapath, gta_images_fileapath, 'GTAV')
    gta_samples = gta_Dataset.getTotalCount()

    # Total training Samples
    total_train_samples = kitti_num_train_samples + ddad_num_train_samples + \
        urbansyn_samples + muad_samples + gta_samples
    print(total_train_samples, ': Total training samples')

    # Total validation samples
    total_val_samples = kitti_num_val_samples + ddad_num_val_samples
    print(total_val_samples, ': total validation samples')

    
    # Pre-trained model checkpoint path
    root_path = \
        '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/2025_02_09/model/'
    checkpoint_path = root_path + 'iter_759999_epoch_16_step_28048.pth'    
    
    # Trainer Class
    trainer = Scene3DTrainer(checkpoint_path = checkpoint_path, is_pretrained=True)
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 40
    batch_size = 5

    # Test images

    # Epochs
    for epoch in range(16, num_epochs):

        print('Epoch: ', epoch + 1)

        # Iterators for datasets
        kitti_count = 0
        ddad_count = 0
        urbansyn_count = 0
        muad_count = 0
        gta_count = 0
        
        is_kitti_complete = False
        is_ddad_complete = False
        is_urbansyn_complete = False
        is_muad_complete = False
        is_gta_complete = False

        is_finetuned = False
        data_list = [] 
        
        data_list.append('URBANSYN')
        data_list.append('MUAD')
        data_list.append('GTAV')
        data_list.append('KITTI')
        data_list.append('DDAD')
        
        random.shuffle(data_list)

        data_list_count = 0

        # Learning Rate schedule            
        if(epoch >= 30):
            trainer.set_learning_rate(0.0000125)

        
        randomlist_kitti = random.sample(range(0, kitti_num_train_samples), kitti_num_train_samples)
        randomlist_ddad = random.sample(range(0, ddad_num_train_samples), ddad_num_train_samples)
        randomlist_urbansyn = random.sample(range(0, urbansyn_samples), urbansyn_samples)
        randomlist_muad = random.sample(range(0, muad_samples), muad_samples)
        randomlist_gta = random.sample(range(0, gta_samples), gta_samples)

        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch

            count += 1

            if(kitti_count == kitti_num_train_samples and \
                is_kitti_complete == False):
                kitti_count = 0
                randomlist_kitti = random.sample(range(0, kitti_num_train_samples), kitti_num_train_samples)
            
            if(ddad_count == ddad_num_train_samples and \
                is_ddad_complete == False):
                ddad_count = 0
                randomlist_ddad = random.sample(range(0, ddad_num_train_samples), ddad_num_train_samples)
            
            if(urbansyn_count == urbansyn_samples and \
                is_urbansyn_complete == False):
                urbansyn_count = 0
                randomlist_urbansyn = random.sample(range(0, urbansyn_samples), urbansyn_samples)
      
  
            if(muad_count == muad_samples and \
                is_muad_complete == False):
                muad_count = 0
                randomlist_muad = random.sample(range(0, muad_samples), muad_samples)

            
            if(gta_count == gta_samples and \
                is_gta_complete == False): 
                gta_count = 0
                randomlist_gta = random.sample(range(0, gta_samples), gta_samples)


            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Dataset sample 
            data_sample = ''
            image = 0
            gt = 0
            validity = 0
            scale_factor = 0

            if(data_list[data_list_count] == 'KITTI' and \
                is_kitti_complete == False):
                image, gt, validity = kitti_Dataset.getItemTrain(randomlist_kitti[kitti_count])
                data_sample = 'KITTI'
                scale_factor = s_kitti
                kitti_count += 1

            if(data_list[data_list_count] == 'DDAD' and \
                is_ddad_complete == False):
                image, gt, validity = ddad_Dataset.getItemTrain(randomlist_ddad[ddad_count])
                data_sample = 'DDAD'
                
                if(ddad_train_cams[randomlist_ddad[ddad_count]] == 'back_camera'):
                    scale_factor = s_ddad_b
                elif(ddad_train_cams[randomlist_ddad[ddad_count]] == 'front_camera'):
                    scale_factor = s_ddad_f

                ddad_count += 1

            if(data_list[data_list_count] == 'URBANSYN' and \
               is_urbansyn_complete == False):
                image, gt, validity = urbansyn_Dataset.getItemAll(randomlist_urbansyn[urbansyn_count])
                data_sample = 'URBANSYN'
                scale_factor = s_urbansyn      
                urbansyn_count += 1

            if(data_list[data_list_count] == 'MUAD' and \
                is_muad_complete == False):
                image, gt, validity = muad_Dataset.getItemAll(randomlist_muad[muad_count])
                scale_factor = s_muad
                data_sample = 'MUAD'
                muad_count += 1

            if(data_list[data_list_count] == 'GTAV' and \
                is_gta_complete == False):
                image, gt, validity = gta_Dataset.getItemAll(randomlist_gta[gta_count])
                data_sample = 'GTAV'      
                scale_factor = s_gta
                gta_count += 1

            # Assign Data
            trainer.set_data(image, gt, validity, scale_factor)
            
            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

            # Run model and calculate loss
            trainer.run_model(data_sample)

            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if((count+1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if((count+1) % 252 == 0):
                trainer.log_loss(log_count, is_finetuned=is_finetuned)
            
            # Logging Image to Tensor Board every 1000 steps
            if((count+1) % 1002 == 0):  
                trainer.save_visualization(log_count)

            
            
            # Save model and run validation on entire validation 
            # dataset after 20000 steps
            if((log_count+1) % 20000 == 0):
                
                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(log_count) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                
                trainer.save_model(model_save_path)

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                # Error
                running_mAE_overall = 0
                running_mAE_kitti = 0
                running_mAE_ddad = 0

                # No gradient calculation
                with torch.no_grad():

                    # KITTI
                    for val_count in range(0, kitti_num_val_samples):
                        image_val, gt_val, validity_val = kitti_Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val, s_kitti)

                        # Accumulating mAE score
                        running_mAE_kitti += mAE
                        running_mAE_overall += mAE

                    # DDAD
                    for val_count in range(0, ddad_num_val_samples):
                        image_val, gt_val, validity_val = ddad_Dataset.getItemVal(val_count)

                        s_ddad = 0
                        if(ddad_val_cams[val_count] == 'back_camera'):
                            s_ddad = s_ddad_b
                        elif(ddad_val_cams[val_count] == 'front_camera'):
                            s_ddad = s_ddad_f

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val, s_ddad)

                        # Accumulating mAE score
                        running_mAE_ddad += mAE
                        running_mAE_overall += mAE


                    # LOGGING
                    # Calculating average loss of complete validation set for
                    # each specific dataset as well as the overall combined dataset
                    avg_mAE_overall = running_mAE_overall/total_val_samples
                    avg_mAE_kitti = running_mAE_kitti/kitti_num_val_samples
                    avg_mAE_ddad = running_mAE_ddad/ddad_num_val_samples

                    
                    # Logging average validation loss to TensorBoard
                    trainer.log_val_mAE(avg_mAE_overall, avg_mAE_kitti, 
                       avg_mAE_ddad, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
            
            data_list_count += 1
            
    trainer.cleanup()
    
    
if __name__ == '__main__':
    main()
# %%
