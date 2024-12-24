#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from scene_seg_trainer import SceneSegTrainer
from argparse import ArgumentParser
import sys
sys.path.append('..')
from data_utils.load_data_scene_seg import LoadDataSceneSeg

def main():

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-r", "--root", dest="root", help="root path to folder where data training data is stored")
    args = parser.parse_args()

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    
    # Root path
    root = args.root

    # Data paths
    # ACDC
    acdc_labels_filepath= root + 'ACDC/gt_masks/'
    acdc_images_filepath = root + 'ACDC/images/'

    # BDD100K
    bdd100k_labels_filepath = root + 'BDD100K/gt_masks/'
    bdd100k_images_filepath = root + 'BDD100K/images/'

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
    _, acdc_num_val_samples = acdc_Dataset.getItemCount()

    # BDD100K - Data Loading
    bdd100k_Dataset = LoadDataSceneSeg(bdd100k_labels_filepath, bdd100k_images_filepath, 'BDD100K')
    bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()

    # IDDAW - Data Loading
    iddaw_Dataset = LoadDataSceneSeg(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
    _, iddaw_num_val_samples = iddaw_Dataset.getItemCount()

    # MUSES - Data Loading
    muses_Dataset = LoadDataSceneSeg(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
    _, muses_num_val_samples = muses_Dataset.getItemCount()

    # Mapillary - Data Loading
    mapillary_Dataset = LoadDataSceneSeg(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
    _, mapillary_num_val_samples = mapillary_Dataset.getItemCount()

    # comma10k - Data Loading
    comma10k_Dataset = LoadDataSceneSeg(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
    _, comma10k_num_val_samples = comma10k_Dataset.getItemCount()

    # Total number of training samples - BDD100K Data was unseen during training
    total_test_samples = bdd100k_num_train_samples + \
    + bdd100k_num_val_samples
    print(total_test_samples, ': total test samples')

    # Total number of validation samples
    total_val_samples = acdc_num_val_samples + \
    + iddaw_num_val_samples + muses_num_val_samples \
    + mapillary_num_val_samples + comma10k_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Trainer Class
    trainer = SceneSegTrainer(checkpoint_path=model_checkpoint_path)
    trainer.zero_grad()
    
    # Setting model to evaluation mode
    trainer.set_eval_mode()

    val_data_list = []
    val_data_list.append('ACDC')
    val_data_list.append('MUSES')
    val_data_list.append('IDDAW')
    val_data_list.append('COMMA10K')
    val_data_list.append('MAPILLARY')

    # Overall Val
    running_IoU_full = 0
    running_IoU_bg = 0
    running_IoU_fg = 0
    running_IoU_rd = 0

    # No gradient calculation
    with torch.no_grad():
        
        for dataset_count in range (0, len(val_data_list)):
            dataset = val_data_list[dataset_count]

            # Dataset specific validation metrics
            dataset_IoU_full = 0
            dataset_IoU_bg = 0
            dataset_IoU_fg = 0
            dataset_IoU_rd = 0

            # Dataset specific validation samples
            val_samples = 0

            if(dataset == 'ACDC'):
                val_samples = acdc_num_val_samples
            elif(dataset == 'MUSES'):
                val_samples = muses_num_val_samples
            elif(dataset == 'IDDAW'):
                val_samples = iddaw_num_val_samples
            elif(dataset == 'COMMA10K'):
                val_samples = comma10k_num_val_samples
            elif(dataset == 'MAPILLARY'):
                val_samples = mapillary_num_val_samples
            
            print('Processing Dataset: ', dataset)

            for val_count in range(0, val_samples):

                image_val = 0
                gt_val = 0

                if(dataset == 'ACDC'):
                    image_val, gt_val, _ = \
                        acdc_Dataset.getItemVal(val_count) 
                elif(dataset == 'MUSES'):
                    image_val, gt_val, _ = \
                        muses_Dataset.getItemVal(val_count) 
                elif(dataset == 'IDDAW'):
                    image_val, gt_val, _ = \
                        iddaw_Dataset.getItemVal(val_count) 
                elif(dataset == 'COMMA10K'):
                    image_val, gt_val, _ = \
                        comma10k_Dataset.getItemVal(val_count) 
                elif(dataset == 'MAPILLARY'):
                    image_val, gt_val, _ = \
                        mapillary_Dataset.getItemVal(val_count) 

                # Run Validation and calculate IoU Score
                IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                    trainer.validate(image_val, gt_val)   
         
                dataset_IoU_full += IoU_score_full
                dataset_IoU_bg += IoU_score_bg
                dataset_IoU_fg += IoU_score_fg
                dataset_IoU_rd += IoU_score_rd

                running_IoU_full += IoU_score_full
                running_IoU_bg += IoU_score_bg
                running_IoU_fg += IoU_score_fg
                running_IoU_rd += IoU_score_rd

            # Dataset specific Validation Metrics
            dataset_IoU_full = dataset_IoU_full/val_samples
            dataset_IoU_bg = dataset_IoU_bg/val_samples
            dataset_IoU_fg = dataset_IoU_fg/val_samples
            dataset_IoU_rd = dataset_IoU_rd/val_samples

            print('-----------------------------------')
            print('Dataset:', dataset, ' Validation Samples: ', val_samples)
            print(dataset, ' IoU Overall ', dataset_IoU_full)
            print(dataset, ' IoU Background ', dataset_IoU_bg)
            print(dataset, ' IoU Foreground ', dataset_IoU_fg)
            print(dataset, ' IoU Road ', dataset_IoU_rd)

        # Cross-Dataset Validation Metrics
        mIoU_full = running_IoU_full/total_val_samples
        mIoU_bg = running_IoU_bg/total_val_samples
        mIoU_fg = running_IoU_fg/total_val_samples
        mIoU_rd = running_IoU_rd/total_val_samples

        print('Overall Validation Set with Total Validation Samples: ', total_val_samples)
        print('Cross Dataset IoU Overall ', mIoU_full)
        print('Cross Dataset IoU Background ', mIoU_bg)
        print('Cross Dataset IoU Foreground ', mIoU_fg)
        print('Cross Dataset IoU Road ', mIoU_rd)
        
        test_IoU_full = 0
        test_IoU_bg = 0
        test_IoU_fg = 0
        test_IoU_rd = 0
        
        # Test set on BDD100K dataset
        for dataset_count in range (0, bdd100k_num_val_samples):
            image_val, gt_val, _ = \
                bdd100k_Dataset.getItemVal(dataset_count) 
            
            # Run Validation and calculate IoU Score
            IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                    trainer.validate(image_val, gt_val)   
            
            test_IoU_full += IoU_score_full
            test_IoU_bg += IoU_score_bg
            test_IoU_fg += IoU_score_fg
            test_IoU_rd += IoU_score_rd

        for dataset_count in range (0, bdd100k_num_train_samples):
            image_val, gt_val, _ = \
                bdd100k_Dataset.getItemTrain(dataset_count) 
            
            # Run Validation and calculate IoU Score
            IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                    trainer.validate(image_val, gt_val)   
            
            test_IoU_full += IoU_score_full
            test_IoU_bg += IoU_score_bg
            test_IoU_fg += IoU_score_fg
            test_IoU_rd += IoU_score_rd

        # Test Dataset Validation Metrics
        mIoU_test_full = test_IoU_full/total_test_samples
        mIoU_test_bg = test_IoU_bg/total_test_samples
        mIoU_test_fg = test_IoU_fg/total_test_samples
        mIoU_test_rd = test_IoU_rd/total_test_samples

        print('BDD100K - Overall Test Set with Total Testing Samples: ', total_test_samples)
        print('Test Dataset IoU Overall ', mIoU_test_full)
        print('Test Dataset IoU Background ', mIoU_test_bg)
        print('Test Dataset IoU Foreground ', mIoU_test_fg)
        print('Test Dataset IoU Road ', mIoU_test_rd)


    

if __name__ == '__main__':
    main()
# %%
