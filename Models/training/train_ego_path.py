#! /usr/bin/env python3
#%%
import os
import torch
import random
import pathlib
from argparse import ArgumentParser
import sys
sys.path.append('../..')
from Models.data_utils.load_data_ego_path import LoadDataEgoPath
from Models.training.ego_path_trainer import EgoPathTrainer

def main():

    # ====================== Parsing input arguments ====================== #

    parser = ArgumentParser()

    parser.add_argument("-r", "--root", dest="root", required=True, \
        help="root path to folder where data training data is stored")
    
    parser.add_argument("-b", "--backbone_path", dest="backbone_path", \
        help="path to SceneSeg *.pth checkpoint file to load pre-trained backbone " \
        "if we are training EgoPath from scratch")
    
    parser.add_argument("-c", "--checkpoint_path", dest="checkpoint_path", \
        help="path to saved EgoPath *.pth checkpoint file for training from saved checkpoint")

    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path", \
        help="root path where pytorch checkpoint file should be saved")
    
    parser.add_argument(
        "-t", "--test_images_save_root_path", 
        dest = "test_images_save_root_path",
        required = False, default = None,
        help = "root path where test images should be saved")


    args = parser.parse_args()

    # ====================== Loading Data ====================== #
    
    # ROOT PATH
    root = args.root

    # MODEL SAVE ROOT PATH
    model_save_root_path = args.model_save_root_path
    
    # BDD100K
    bdd100k_labels_filepath= root + 'BDD100K/drivable_path.json'
    bdd100k_images_filepath = root + 'BDD100K/image/'

    # COMMA2K19
    comma2k19_labels_fileapath = root + 'COMMA2K19/drivable_path.json'
    comma2k19_images_fileapath = root + 'COMMA2K19/image/'

    # CULANE
    culane_labels_fileapath = root + 'CULANE/drivable_path.json'
    culane_images_fileapath = root + 'CULANE/image/'

    # CURVELANES
    curvelanes_labels_fileapath = root + 'CURVELANES/drivable_path.json'
    curvelanes_images_fileapath = root + 'CURVELANES/image/'

    # ROADWORK
    roadwork_labels_fileapath = root + 'ROADWORK/drivable_path.json'
    roadwork_images_fileapath = root + 'ROADWORK/image/'

    # TUSIMPLE
    tusimple_labels_filepath = root + 'TUSIMPLE/drivable_path.json'
    tusimple_images_filepath = root + 'TUSIMPLE/image/'

    # TEST
    if (args.test_images_save_root_path):
        test_images_filepath = root + 'TEST/'
        test_images_list = sorted([
                f for f in pathlib.Path(test_images_filepath).glob("*.png")
            ])
        test_images_save_root_path = args.test_images_save_root_path
    else:
        test_images_save_root_path = None

    # TEST
    #### to do ####

    # BDD100K - Data Loading
    bdd100k_Dataset = LoadDataEgoPath(bdd100k_labels_filepath, bdd100k_images_filepath, 'BDD100K')
    bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()
    bdd100k_sample_list = list(range(0, bdd100k_num_train_samples))
    random.shuffle(bdd100k_sample_list)

    # COMMA2K19 - Data Loading
    comma2k19_Dataset = LoadDataEgoPath(comma2k19_labels_fileapath, comma2k19_images_fileapath, 'COMMA2K19')
    comma2k19_num_train_samples, comma2k19_num_val_samples = comma2k19_Dataset.getItemCount()
    comma2k19_sample_list = list(range(0, comma2k19_num_train_samples))
    random.shuffle(comma2k19_sample_list)

    # CULANE - Data Loading
    culane_Dataset = LoadDataEgoPath(culane_labels_fileapath, culane_images_fileapath, 'CULANE')
    culane_num_train_samples, culane_num_val_samples = culane_Dataset.getItemCount()
    culane_sample_list = list(range(0, culane_num_train_samples))
    random.shuffle(culane_sample_list)

    # CURVELANES - Data Loading
    curvelanes_Dataset = LoadDataEgoPath(curvelanes_labels_fileapath, curvelanes_images_fileapath, 'CURVELANES')
    curvelanes_num_train_samples, curvelanes_num_val_samples = curvelanes_Dataset.getItemCount()
    curvelanes_sample_list = list(range(0, curvelanes_num_train_samples))
    random.shuffle(curvelanes_sample_list)

    # ROADWORK - Data Loading
    roadwork_Dataset = LoadDataEgoPath(roadwork_labels_fileapath, roadwork_images_fileapath, 'ROADWORK')
    roadwork_num_train_samples, roadwork_num_val_samples = roadwork_Dataset.getItemCount()
    roadwork_sample_list = list(range(0, roadwork_num_train_samples))
    random.shuffle(roadwork_sample_list)

    # TUSIMPLE - Data Loading
    tusimple_Dataset = LoadDataEgoPath(tusimple_labels_filepath, tusimple_images_filepath, 'TUSIMPLE')
    tusimple_num_train_samples, tusimple_num_val_samples = tusimple_Dataset.getItemCount()
    tusimple_sample_list = list(range(0, tusimple_num_train_samples))
    random.shuffle(tusimple_sample_list)

    # ==================== Training/Val Samples ==================== #

    # Total number of training samples
    total_train_samples = bdd100k_num_train_samples + \
    + comma2k19_num_train_samples + culane_num_train_samples \
    + curvelanes_num_train_samples + roadwork_num_train_samples \
    + tusimple_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = bdd100k_num_val_samples + \
    + comma2k19_num_val_samples + culane_num_val_samples \
    + curvelanes_num_val_samples + roadwork_num_val_samples \
    + tusimple_num_val_samples
    print(total_val_samples, ': total validation samples')

    # ====================== Training ====================== #

    # Trainer instance
    trainer = 0

    # Loading from checkpoint or training from scratch
    backbone_path = args.backbone_path 
    checkpoint_path = args.checkpoint_path

    if(backbone_path and not checkpoint_path):
        trainer = EgoPathTrainer(pretrained_checkpoint_path=backbone_path)
    elif(checkpoint_path and not backbone_path):
        trainer = EgoPathTrainer(checkpoint_path=checkpoint_path, is_pretrained=True)
    elif(not checkpoint_path and not backbone_path):
        raise ValueError('No checkpoint file found - Please ensure that you pass in either' \
                          ' a saved EgoPath checkpoint file or SceneSeg checkpoint to load' \
                          ' the backbone weights')
    else:
        raise ValueError('Both EgoPath checkpoint and SceneSeg checkpoint file provided - ' \
                        ' please provide either the EgoPath checkponit to continue training' \
                        ' from a saved checkpoint, or the SceneSeg checkpoint file to train' \
                        ' EgoPath from scratch')

    # Zero gradients
    trainer.zero_grad()
    
    # CONSTANTS - (DO NOT CHANGE) - Training loop parameters
    NUM_EPOCHS = 20
    LOGSTEP_LOSS = 250
    LOGSTEP_VIS = 1000
    LOGSTEP_MODEL = 20000


    ####################################################################
    ###################### MODIFIABLE PARAMETERS #######################
    # You can adjust the SCALE FACTORS, GRAD_LOSS_TYPE, DATA_SAMPLNG_SCHEME 
    # and BATCH_SIZE_DECAY during training

    # ========================= SCALE FACTORS ========================= #
    # These scale factors impact the relative weight of different
    # loss terms in calculating the overall loss. A scale factor
    # value of 0.0 means that this loss is ignored, 1.0 means that
    # the loss is not scaled, and any other number applies a simple
    # scaling to increase or decrease the contribution of that specific
    # loss towards the overall loss

    ENDPOINT_LOSS_SCALE_FACTOR = 1.0          
    MIDPOINT_LOSS_SCALE_FACTOR = 1.0
    GRADIENT_LOSS_SCALE_FACTOR = 1.0

    # Set training loss term scale factors
    trainer.set_loss_scale_factors(ENDPOINT_LOSS_SCALE_FACTOR, 
            MIDPOINT_LOSS_SCALE_FACTOR, GRADIENT_LOSS_SCALE_FACTOR)
    
    print('ENDPOINT_LOSS_SCALE_FACTOR: ', ENDPOINT_LOSS_SCALE_FACTOR)
    print('MIDPOINT_LOSS_SCALE_FACTOR: ', MIDPOINT_LOSS_SCALE_FACTOR)
    print('GRADIENT_LOSS_SCALE_FACTOR: ', GRADIENT_LOSS_SCALE_FACTOR)
    # ======================== GRAD_LOSS_TYPE ======================== #
    # There are two types of gradients loss, and either can be selected.
    # One option is 'NUMERICAL' which calculates the gradient through
    # the tangent angle between consecutive pairs of points along the
    # curve. The second option is 'ANALYTICAL' which uses the equation
    # of the curve to calculate the true mathematical gradient from
    # the curve's partial dervivatives

    GRAD_LOSS_TYPE = 'NUMERICAL' # NUMERICAL or ANALYTICAL
    trainer.set_gradient_loss_type(GRAD_LOSS_TYPE)

    print('GRAD_LOSS_TYPE: ', GRAD_LOSS_TYPE)
    # ====================== DATA_SAMPLING_SCHEME ====================== #
    # There are two data sampling schemes. The 'EQUAL' data sampling scheme
    # ensures that in each batch, we have an equal representation of samples
    # from each specific dataset. This schemes over-fits the network on 
    # smaller and underepresented datasets. The second sampling scheme is
    # 'CONCATENATE', in which the data is sampled randomly and the network
    # only sees each image from each dataset once in an epoch

    DATA_SAMPLING_SCHEME = 'EQUAL' # EQUAL or CONCATENATE

    print('DATA_SAMPLING_SCHEME: ', DATA_SAMPLING_SCHEME)
    # ======================== BATCH_SIZE_SCHEME ======================== #
    # There are three type of BATCH_SIZE_SCHEME, the 'CONSTANT' batch size
    # scheme sets a constant, fixed batch size value of 24 throughout training.
    # The 'SLOW_DECAY' batch size scheme reduces the batch size during training,
    # helping the model escape from local minima. The 'FAST_DECAY' batch size
    # scheme decays the batch size faster, and may help with quicker model
    # convergence.

    BATCH_SIZE_SCHEME = 'FAST_DECAY' # FAST_DECAY or SLOW_DECAY or CONSTANT

    print('BATCH_SIZE_SCHEME: ', BATCH_SIZE_SCHEME)
    ####################################################################

    # Datasets list
    data_list = []
    data_list.append('BDD100K')
    data_list.append('COMMA2K19')
    data_list.append('CULANE')
    data_list.append('CURVELANES')
    data_list.append('ROADWORK')
    data_list.append('TUSIMPLE')

    # Initialize batch_size variable
    batch_size = 0
    
    # Running through epochs
    for epoch in range(0, NUM_EPOCHS):

        print('EPOCH: ', epoch)

        if(BATCH_SIZE_SCHEME == 'CONSTANT'):
            batch_size = 24
        elif(BATCH_SIZE_SCHEME == 'FAST_DECAY'):
            if(epoch == 0):
                batch_size = 24
            elif(epoch >= 2 and epoch < 4):
                batch_size = 12
            elif(epoch >= 4 and epoch < 6):
                batch_size = 6
            elif(epoch >= 6 and epoch < 8):
                batch_size = 3
            elif(epoch >= 8 and epoch < 10):
                batch_size = 2
            elif(epoch >= 10):
                batch_size = 1
        elif(BATCH_SIZE_SCHEME == 'SLOW_DECAY'):
            if(epoch == 0):
                batch_size = 24
            elif(epoch >= 2 and epoch < 6):
                batch_size = 12
            elif(epoch >= 6 and epoch < 10):
                batch_size = 6
            elif(epoch >= 10 and epoch < 14):
                batch_size = 3
            elif(epoch >= 14 and epoch < 18):
                batch_size = 2
            elif(epoch >= 18):
                batch_size = 1
        else:
            raise ValueError('Please speficy BATCH_SIZE_SCHEME as either' \
                ' CONSTANT or FAST_DECAY or SLOW_DECAY')

        # Learning Rate Schedule
        if(epoch >= 6 and epoch < 12):
            trainer.set_learning_rate(0.00005)
        elif(epoch >= 12 and epoch < 18):
            trainer.set_learning_rate(0.000025)
        elif(epoch >= 18):
            trainer.set_learning_rate(0.0000125)

        # Shuffle overall data list at start of epoch and reset data list counter
        random.shuffle(data_list)
        data_list_count = 0

        # Iterators for datasets
        bdd100k_count = 0
        comma2k19_count = 0
        culane_count = 0
        curvelanes_count = 0
        roadwork_count = 0
        tusimple_count = 0

        # Flags for whether all samples from a dataset have been read during the epoch
        is_bdd100k_complete = False
        is_comma2k19_complete = False
        is_culane_complete = False
        is_curvelanes_complete = False
        is_roadwork_complete = False
        is_tusimple_complete = False

        # Checking data sampling scheme
        if(DATA_SAMPLING_SCHEME != 'EQUAL' and DATA_SAMPLING_SCHEME != 'CONCATENATE'):
            raise ValueError('Please speficy DATA_SAMPLING_SCHEME as either' \
                ' EQUAL or CONCATENATE')

        # Data sample counter
        count = 0

        # Loop through data
        while(True):

            # Log count
            count += 1
            log_count = count + total_train_samples*epoch

            # Reset iterators and shuffle individual datasets
            # based on data sampling scheme
            if(bdd100k_count == bdd100k_num_train_samples):
                
                if(DATA_SAMPLING_SCHEME == 'EQUAL'):
                    bdd100k_count = 0
                    random.shuffle(bdd100k_sample_list)
                elif(DATA_SAMPLING_SCHEME == 'CONCATENATE' and 
                        is_bdd100k_complete == False):
                    data_list.remove("BDD100K")

                is_bdd100k_complete = True
            
            if(comma2k19_count == comma2k19_num_train_samples):
                
                if(DATA_SAMPLING_SCHEME == 'EQUAL'):
                    comma2k19_count = 0
                    random.shuffle(comma2k19_sample_list)
                elif(DATA_SAMPLING_SCHEME == 'CONCATENATE' and 
                        is_comma2k19_complete == False):
                    data_list.remove('COMMA2K19')

                is_comma2k19_complete = True
            
            if(culane_count == culane_num_train_samples):
                
                if(DATA_SAMPLING_SCHEME == 'EQUAL'):
                    culane_count = 0
                    random.shuffle(culane_sample_list)
                elif(DATA_SAMPLING_SCHEME == 'CONCATENATE' and 
                        is_culane_complete == False):
                    data_list.remove('CULANE')    

                is_culane_complete = True

            if(curvelanes_count == curvelanes_num_train_samples):
                
                curvelanes_count = 0
                random.shuffle(curvelanes_sample_list)

                is_curvelanes_complete = True

            if(roadwork_count == roadwork_num_train_samples):
                
                if(DATA_SAMPLING_SCHEME == 'EQUAL'):
                    roadwork_count = 0
                    random.shuffle(roadwork_sample_list)
                elif(DATA_SAMPLING_SCHEME == 'CONCATENATE' and
                        is_roadwork_complete == False):
                    data_list.remove('ROADWORK')

                is_roadwork_complete = True

            if(tusimple_count == tusimple_num_train_samples):
                
                if(DATA_SAMPLING_SCHEME == 'EQUAL'):
                    tusimple_count = 0
                    random.shuffle(tusimple_sample_list)
                elif(DATA_SAMPLING_SCHEME == 'CONCATENATE' and
                        is_tusimple_complete == False):
                    data_list.remove('TUSIMPLE')

                is_tusimple_complete = True

            # If we have looped through each dataset at least once - restart the epoch
            if(is_bdd100k_complete and is_comma2k19_complete and is_culane_complete
               and is_curvelanes_complete and is_roadwork_complete and is_tusimple_complete):
                break
            
            # Reset the data list count if out of range
            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Get data depending on which dataset we are processing
            image = 0
            gt = 0
            is_valid = True

            if(data_list[data_list_count] == 'BDD100K'):
                image, gt, is_valid = bdd100k_Dataset.getItem(bdd100k_sample_list[bdd100k_count], is_train=True)
                bdd100k_count += 1

            if(data_list[data_list_count] == 'COMMA2K19'):
                image, gt, is_valid = comma2k19_Dataset.getItem(comma2k19_sample_list[comma2k19_count], is_train=True)
                comma2k19_count += 1

            if(data_list[data_list_count] == 'CULANE'):
                image, gt, is_valid = culane_Dataset.getItem(culane_sample_list[culane_count], is_train=True)
                culane_count += 1

            if(data_list[data_list_count] == 'CURVELANES'):
                image, gt, is_valid = curvelanes_Dataset.getItem(curvelanes_sample_list[curvelanes_count], is_train=True)
                curvelanes_count += 1

            if(data_list[data_list_count] == 'ROADWORK'):
                image, gt, is_valid = roadwork_Dataset.getItem(roadwork_sample_list[roadwork_count], is_train=True)
                roadwork_count += 1

            if(data_list[data_list_count] == 'TUSIMPLE'):
                image, gt, is_valid = tusimple_Dataset.getItem(tusimple_sample_list[tusimple_count], is_train=True)
                tusimple_count += 1
            
            # If the data is valid
            if(is_valid):

                # Assign data
                trainer.set_data(image, gt)
                
                # Augment image
                trainer.apply_augmentations(is_train = True)
                
                # Converting to tensor and loading
                trainer.load_data()

                # Run model and get loss
                trainer.run_model()
                
                # Gradient accumulation
                trainer.loss_backward()

                # Simulating batch size through gradient accumulation
                if((count+1) % batch_size == 0):
                    trainer.run_optimizer()

                # Logging loss to Tensor Board
                if((count+1) % LOGSTEP_LOSS == 0):
                    trainer.log_loss(log_count+1)
                
                # Logging Visualization to Tensor Board
                if((count+1) % LOGSTEP_VIS == 0):  
                    trainer.save_visualization(log_count+1)
                
                # Save model and run Validation on entire validation dataset
                if ((count+1) % LOGSTEP_MODEL == 0):
                    
                    print('\n')
                    print('Iteration:', count+1)
                    print('----- Saving Model -----')

                    # Save model
                    model_save_path = os.path.join(
                        model_save_root_path,
                        f"iter_{log_count+1}_epoch_{epoch}_step_{count+1}.pth"
                    )
                    trainer.save_model(model_save_path)
                    
                    # Set model to eval mode
                    trainer.set_eval_mode()

                    # Running test
                    if (test_images_save_root_path):
                        print('----- Running Testing -----')
                        for i in range(0, len(test_images_list)):
                            
                            test_image_save_path = os.path.join(
                                test_images_save_root_path,
                                f"iter_{log_count+1}_epoch_{epoch}_step_{count+1}_{i}.png"
                            )

                            test_image_path = str(test_images_list[i])
                            trainer.test(test_image_path, test_image_save_path)

                    # Validation metrics for each dataset
                    val_bdd100k_running = 0
                    num_val_bdd100k_samples = 0

                    val_comma2k19_running = 0
                    num_val_comma2k19_samples = 0

                    val_culane_running = 0
                    num_val_culane_samples = 0

                    val_curvelanes_running = 0
                    num_val_curvelanes_samples = 0

                    val_roadwork_running = 0
                    num_val_roadwork_samples = 0

                    val_tusimple_running = 0
                    num_val_tusimple_samples = 0

                    # Temporarily disable gradient computation for backpropagation
                    with torch.no_grad():

                        print('----- Running validation calculation -----')
                        # Compute val loss per dataset
                        
                        # BDD100K
                        for val_count in range(0, bdd100k_num_val_samples):
                            image, gt, is_valid = bdd100k_Dataset.getItem(val_count, is_train=False)

                            if(is_valid):
                                num_val_bdd100k_samples += 1
                                val_metric = trainer.validate(image, gt)
                                val_bdd100k_running = val_bdd100k_running + val_metric
                        
                        # COMMA2K19
                        for val_count in range(0, comma2k19_num_val_samples):
                            image, gt, is_valid = comma2k19_Dataset.getItem(val_count, is_train=False)

                            if(is_valid):
                                num_val_comma2k19_samples += 1
                                val_metric = trainer.validate(image, gt)
                                val_comma2k19_running = val_comma2k19_running + val_metric
                        
                        # CULANE
                        for val_count in range(0, culane_num_val_samples):
                            image, gt, is_valid = culane_Dataset.getItem(val_count, is_train=False)

                            if(is_valid):
                                num_val_culane_samples += 1
                                val_metric = trainer.validate(image, gt)
                                val_culane_running = val_culane_running + val_metric
                        
                        # CURVELANES
                        for val_count in range(0, curvelanes_num_val_samples):
                            image, gt, is_valid = curvelanes_Dataset.getItem(val_count, is_train=False)

                            if(is_valid):
                                num_val_curvelanes_samples += 1
                                val_metric = trainer.validate(image, gt)
                                val_curvelanes_running = val_curvelanes_running + val_metric
                        
                        # ROADWORK
                        for val_count in range(0, roadwork_num_val_samples):
                            image, gt, is_valid = roadwork_Dataset.getItem(val_count, is_train=False)

                            if(is_valid):
                                num_val_roadwork_samples += 1
                                val_metric = trainer.validate(image, gt)
                                val_roadwork_running = val_roadwork_running + val_metric
                        
                        # TUSIMPLE
                        for val_count in range(0, tusimple_num_val_samples):
                            image, gt, is_valid = tusimple_Dataset.getItem(val_count, is_train=False)

                            if(is_valid):
                                num_val_tusimple_samples += 1
                                val_metric = trainer.validate(image, gt)
                                val_tusimple_running = val_tusimple_running + val_metric
                        
                        # Calculate final validation scores for network on each dataset
                        # as well as overall validation score - A lower score is better
                        bdd100k_val_score = val_bdd100k_running/num_val_bdd100k_samples
                        comma2k19_val_score = val_comma2k19_running/num_val_comma2k19_samples
                        culane_val_score = val_culane_running/num_val_culane_samples
                        curvelanes_val_score = val_curvelanes_running/num_val_curvelanes_samples
                        roadwork_val_score = val_roadwork_running/num_val_roadwork_samples
                        tusimple_val_score = val_tusimple_running/num_val_tusimple_samples

                        # Ovearll validation metric
                        val_overall_running = val_bdd100k_running + val_comma2k19_running + \
                            val_culane_running + val_curvelanes_running + val_roadwork_running + \
                            val_tusimple_running
                        
                        num_val_overall_samples = num_val_bdd100k_samples + num_val_comma2k19_samples + \
                            num_val_culane_samples + num_val_curvelanes_samples + num_val_roadwork_samples + \
                            num_val_tusimple_samples
                        
                        overall_validation_score = val_overall_running/num_val_overall_samples
                      
                        print('---------- Complete - Validation Scores ----------')
                        print('BDD100K: ', bdd100k_val_score)
                        print('COMM2K19: ', comma2k19_val_score)
                        print('CULANE: ', culane_val_score)
                        print('CURVELANES: ', curvelanes_val_score)
                        print('ROADWORK: ', roadwork_val_score)
                        print('TUSIMPLE: ', tusimple_val_score)
                        print('OVERALL:', overall_validation_score)
                        print('\n')

                        # Logging average metrics
                        trainer.log_validation(log_count, bdd100k_val_score, comma2k19_val_score,
                            culane_val_score, curvelanes_val_score, roadwork_val_score,
                            tusimple_val_score, overall_validation_score)

                    # Switch back to training
                    print('----- Continuing with training -----')
                    trainer.set_train_mode()
                
                data_list_count += 1

    print('----- Training Completed -----')
    trainer.cleanup()
    

if __name__ == "__main__":
    main()

#%%