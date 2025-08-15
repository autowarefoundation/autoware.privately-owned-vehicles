#%%
#! /usr/bin/env python3

import os
import random
import torch
from PIL import Image
from typing import Literal, get_args
import sys
sys.path.append('../..')
from Models.data_utils.load_data_auto_steer import LoadDataAutoSteer
from Models.training.auto_steer_trainer import AutoSteerTrainer

# Currently limiting to available datasets only. Will unlock eventually
VALID_DATASET_LITERALS = Literal["TUSIMPLE"] #Literal["TUSIMPLE", "CULANE"]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))

BEV_JSON_PATH = "drivable_path_bev.json"
BEV_IMG_PATH = "image_bev"
BEV_VIS_PATH = "visualization_bev"
PERSPECTIVE_IMG_PATH = "image"
PERSPECTIVE_VIS_PATH = "visualization"


def main():

   
    # ====================== Loading datasets ====================== #

    # Root
    ROOT_PATH = '/home/zain/Autoware/Data/AutoSteer/'#args.root

    # Model save root path
    MODEL_SAVE_ROOT_PATH = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/AutoSteer/models/' #args.model_save_root_path
    if (not os.path.exists(MODEL_SAVE_ROOT_PATH)):
        os.makedirs(MODEL_SAVE_ROOT_PATH)

    # Visualizations save root path
    VIS_SAVE_ROOT_PATH = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/AutoSteer/figures/'
    if (not os.path.exists(VIS_SAVE_ROOT_PATH)):
        os.makedirs(VIS_SAVE_ROOT_PATH)

    # Init metadata for datasets
    msdict = {}
    for dataset in VALID_DATASET_LIST:
        msdict[dataset] = {
            "path_labels"   : os.path.join(ROOT_PATH, dataset, BEV_JSON_PATH),
            "path_images"   : os.path.join(ROOT_PATH, dataset, BEV_IMG_PATH),
            "path_perspective_vis" : os.path.join(ROOT_PATH, dataset, PERSPECTIVE_VIS_PATH),
            "path_perspective_image": os.path.join(ROOT_PATH, dataset, PERSPECTIVE_IMG_PATH),
            "path_bev_vis" : os.path.join(ROOT_PATH, dataset, BEV_VIS_PATH)
        }

    # Deal with TEST dataset
    #if (args.test_images_save_root_path):
    #    msdict["TEST"] = {
    #        "list_images" : sorted([
    #            f for f in pathlib.Path(
    #                os.path.join(ROOT_PATH, "TEST")
    #            ).glob("*.png")
    #        ]),
    #        "path_test_save" : args.test_images_save_root_path
    #    }

    # Load datasets
    for dataset in VALID_DATASET_LIST:
        this_dataset_loader = LoadDataAutoSteer(
            labels_filepath = msdict[dataset]["path_labels"],
            images_filepath = msdict[dataset]["path_images"],
            dataset = dataset
        )
        N_trains, N_vals = this_dataset_loader.getItemCount()
        random_sample_list = random.sample(list(range(0, N_trains)), N_trains)

        msdict[dataset]["loader"] = this_dataset_loader
        msdict[dataset]["N_trains"] = N_trains
        msdict[dataset]["N_vals"] = N_vals
        msdict[dataset]["sample_list"] = random_sample_list

        print(f"LOADED: {dataset} with {N_trains} train samples, {N_vals} val samples.")

    # All datasets - stats
    msdict["Nsum_trains"] = sum([
        msdict[dataset]["N_trains"]
        for dataset in VALID_DATASET_LIST
    ])
    print(f"Total train samples: {msdict['Nsum_trains']}")

    msdict["Nsum_vals"] = sum([
        msdict[dataset]["N_vals"]
        for dataset in VALID_DATASET_LIST
    ])
    print(f"Total val samples: {msdict['Nsum_vals']}")

    # ====================== Training params ====================== #

    # Trainer instance
    trainer = None

    CHECKPOINT_PATH = None #args.checkpoint_path
    if (CHECKPOINT_PATH != None):
        trainer = AutoSteerTrainer(checkpoint_path = CHECKPOINT_PATH)    
    else:
        trainer = AutoSteerTrainer()
    
    # Zero gradients
    trainer.zero_grad()
    
    # Training loop parameters
    NUM_EPOCHS = 50
    LOGSTEP_LOSS = 50
    LOGSTEP_VIS = 100
    LOGSTEP_MODEL = 5000

    # Val visualization param
    N_VALVIS = 25

    
    # ========================= Main training loop ========================= #
    print('Beginning Training')

    # Batch Size
    batch_size = 8

    for epoch in range(0, NUM_EPOCHS):

        data_list = VALID_DATASET_LIST.copy()
        print(f"EPOCH : {epoch}")

        # Batch Size Schedule
        if (epoch > 2 and epoch <= 4):
            batch_size = 6
        elif (epoch > 4 and epoch <= 6):
            batch_size = 4
        elif (epoch > 8):
            batch_size = 2
      
        # Learning Rate Schedule
        if(epoch <= 4):
            trainer.set_learning_rate(0.0005)
        elif(epoch > 4 and epoch < 8):
            trainer.set_learning_rate(0.0001)
        elif(epoch > 8):
            trainer.set_learning_rate(0.00001)

        # Augmentation Schedule
        apply_augmentation = True
        if (epoch > 35):
            apply_augmentation = False

        # Shuffle overall data list at start of epoch
        random.shuffle(data_list)
        msdict["data_list_count"] = 0
        
        # Reset all data counters
        msdict["sample_counter"] = 0
        for dataset in VALID_DATASET_LIST:
            msdict[dataset]["iter"] = 0
            msdict[dataset]["completed"] = False

        # Loop through data
        while (True):

            # Log count
            msdict["sample_counter"] = msdict["sample_counter"] + 1
            msdict["log_counter"] = (
                msdict["sample_counter"] + \
                msdict["Nsum_trains"] * epoch
            )

            # Reset iterators and shuffle individual datasets
            # based on data sampling scheme
            for dataset in VALID_DATASET_LIST:
                N_trains = msdict[dataset]["N_trains"]
                if (msdict[dataset]["iter"] == N_trains-1):
                    if ((msdict[dataset]["completed"] == False)):
                        data_list.remove(dataset)
                    msdict[dataset]["completed"] = True
            
            # If we have looped through each dataset at least once - restart the epoch
            if (all([
                msdict[dataset]["completed"]
                for dataset in VALID_DATASET_LIST
            ])):
                break

            # Reset the data list count if out of range
            if (msdict["data_list_count"] >= len(data_list)):
                msdict["data_list_count"] = 0

            # Fetch data from current processed dataset
            frame_id = 0
            bev_image = None
            homotrans_mat = []
            bev_egopath = []
            reproj_egopath = []
            bev_egoleft = []
            reproj_egoleft = []
            bev_egoright = []
            reproj_egoright = []
           
            current_dataset = data_list[msdict["data_list_count"]]
            current_dataset_iter = msdict[current_dataset]["iter"]
            [   frame_id, bev_image,
                homotrans_mat,
                bev_egopath, reproj_egopath,
                bev_egoleft, reproj_egoleft,
                bev_egoright, reproj_egoright,
            ] = msdict[current_dataset]["loader"].getItem(
                msdict[current_dataset]["sample_list"][current_dataset_iter],
                is_train = True
            )
            msdict[current_dataset]["iter"] = current_dataset_iter + 1

            # Perspective image
            perspective_image = Image.open(
                os.path.join(
                    msdict[current_dataset]["path_perspective_image"],
                    f"{frame_id}.png"
                )
            ).convert("RGB")
            
            # BEV visualization
            bev_vis = Image.open(
                os.path.join(
                    msdict[current_dataset]["path_bev_vis"],
                    f"{frame_id}.jpg"
                )
            ).convert("RGB")
          
            # Assign data
            trainer.set_data(homotrans_mat, bev_image, perspective_image, \
                bev_egopath, bev_egoleft, bev_egoright, reproj_egopath, \
                reproj_egoleft, reproj_egoright)
            
            # Augment image
            trainer.apply_augmentations(apply_augmentation)
            
            # Converting to tensor and loading
            trainer.load_data()
            
            # Run model and calculate loss
            trainer.run_model()
            
            # Gradient accumulation
            trainer.loss_backward()
            
            # Simulating batch size through gradient accumulation
            if ((msdict["sample_counter"] + 1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board
            if ((msdict["sample_counter"] + 1) % LOGSTEP_LOSS == 0):
                trainer.log_loss(msdict["log_counter"] + 1)
            
            # Logging Visualization to Tensor Board
            if((msdict["sample_counter"] + 1) % LOGSTEP_VIS == 0):  
                trainer.save_visualization(msdict["log_counter"] + 1, bev_vis, is_train=True)
            
            # Save model and run Validation on entire validation dataset
            if ((msdict["sample_counter"] + 1) % LOGSTEP_MODEL == 0):
                
                print(f"\nIteration: {msdict['log_counter'] + 1}")
                print("================ Saving Model ================")

                # Save model
                model_save_path = os.path.join(
                    MODEL_SAVE_ROOT_PATH,
                    f"iter_{msdict['log_counter'] + 1}_epoch_{epoch}_step_{msdict['sample_counter'] + 1}.pth"
                )
                trainer.save_model(model_save_path)
                
                # Set model to eval mode
                trainer.set_eval_mode()

                # Validation metrics for each dataset
                for dataset in VALID_DATASET_LIST:

                    msdict[dataset]["num_val_samples"] = 0
                    msdict[dataset]["total_running"] = 0

                # Temporarily disable gradient computation for backpropagation
                with torch.no_grad():

                    print("================ Running validation calculation ================")
                    
                    # Compute val loss per dataset
                    for dataset in VALID_DATASET_LIST:
                        for val_count in range(0, msdict[dataset]["N_vals"]):

                            # Fetch data
                            [   frame_id, bev_image,
                                homotrans_mat,
                                bev_egopath, reproj_egopath,
                                bev_egoleft, reproj_egoleft,
                                bev_egoright, reproj_egoright,
                            ] = msdict[dataset]["loader"].getItem(
                                val_count,
                                is_train = False
                            )
                            msdict[dataset]["num_val_samples"] = msdict[dataset]["num_val_samples"] + 1
                            
                            # BEV
                            perspective_image = Image.open(
                                os.path.join(
                                    msdict[dataset]["path_perspective_image"],
                                    f"{frame_id}.png"
                                )
                            ).convert("RGB")

                            # BEV visualization
                            bev_vis = Image.open(
                                os.path.join(
                                    msdict[dataset]["path_bev_vis"],
                                    f"{frame_id}.jpg"
                                )
                            ).convert("RGB")

                            # Assign data
                            trainer.set_data(homotrans_mat, bev_image, perspective_image, \
                                bev_egopath, bev_egoleft, bev_egoright, reproj_egopath, \
                                reproj_egoleft, reproj_egoright)
                            
                            # Augment image
                            trainer.apply_augmentations(False)
                            
                            # Converting to tensor and loading
                            trainer.load_data()
                            
                            # Run model and calculate loss
                            trainer.run_model()

                            # Get running total of loss value
                            msdict[dataset]["total_running"] += trainer.get_total_loss()

                            # Save visualization to Tensorboard
                            if(val_count < N_VALVIS): 
                                vis_path = VIS_SAVE_ROOT_PATH + dataset + '_epoch_'+ str(epoch) + '_step_' + \
                                    str(msdict["log_counter"] + 1) + '_image_' + str(frame_id)
                                trainer.save_visualization(msdict["log_counter"] + 1 + val_count, bev_vis, vis_path, is_train=False)


                    # Calculate final validation scores for network on each dataset
                    # as well as overall validation score - A lower score is better

                    # Overall validation score across datasets
                    overall_val_score = 0

                    # Calculating dataset specific validation metrics
                    for dataset in VALID_DATASET_LIST:

                        validation_loss_dataset_total =  msdict[dataset]["total_running"] / msdict[dataset]["num_val_samples"]
                        overall_val_score += validation_loss_dataset_total
                        print("DATASET :", dataset, " VAL SCORE : ", validation_loss_dataset_total)

                        # Logging validation metric for each dataset
                        trainer.log_validation_dataset(dataset, validation_loss_dataset_total, msdict["log_counter"] + 1)
                    
                    overall_val_score = overall_val_score/len(VALID_DATASET_LIST)
                    print("OVERALL VAL SCORE :", overall_val_score)

                    # Logging average metric overall across all datasets
                    trainer.log_validation_overall(overall_val_score, msdict["log_counter"] + 1)
                        
                # Switch back to training
                print("================ Continuing with training ================")
                trainer.set_train_mode()
            
            msdict["data_list_count"] = msdict["data_list_count"] + 1
            

if (__name__ == "__main__"):
    main()
#%%