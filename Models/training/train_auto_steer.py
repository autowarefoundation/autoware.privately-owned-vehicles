"""
# Main train loop for AutoSteer
# Responsible for getting data via the AutoSteer data loader, passing the data to the AutoSteer trainer class
# Running the main train loop over multiple epochs
# Does simulation of batch size (we use a batch size of one)
# Saving the model checkpoints
# Colating the validation metrics
"""

# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import torch
import random
from argparse import ArgumentParser
import sys
import torchvision.transforms as T
import numpy as np

sys.path.append('..')
from Models.data_utils.load_data_auto_steer import LoadDataAutoSteer
from Models.training.auto_steer_trainer import AutoSteerTrainer


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path",
                        help="root path where pytorch checkpoint file should be saved")
    parser.add_argument("-m", "--pretrained_checkpoint_path", dest="pretrained_checkpoint_path",
                        help="path to EgoLanes weights file for pre-trained backbone")
    parser.add_argument("-c", "--checkpoint_path", dest="checkpoint_path",
                        help="path to Scene3D weights file for training from saved checkpoint")
    parser.add_argument('-t', "--test_images_save_path", dest="test_images_save_path",
                        help="path to where visualizations from inference on test images are saved")
    parser.add_argument("-r", "--root", dest="root", help="root path to folder where data training data is stored")
    parser.add_argument('-l', "--load_from_save", action='store_true',
                        help="flag for whether model is being loaded from a EgoLanes checkpoint file")
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path

    # Test data
    test_images = root + 'test/'
    test_images_save_path = args.test_images_save_path

    # AutoSteer - Data Loading
    transform = T.Compose([
        T.Lambda(lambda img: img.crop((0, 420, img.width, img.height))),  # left, top, right, bottom
        T.Resize((320, 640)),
        T.Lambda(lambda img: np.array(img)),
    ])
    auto_steer_dataset = LoadDataAutoSteer(root, transform=transform)
    total_train_samples, total_val_samples = len(auto_steer_dataset.train), len(auto_steer_dataset.val)
    print(total_train_samples, ': total training samples')
    print(total_val_samples, ': total validation samples')

    # Load from checkpoint
    load_from_checkpoint = False
    if (args.load_from_save):
        load_from_checkpoint = True

    # Pre-trained model checkpoint path
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    checkpoint_path = args.pretrained_checkpoint_path

    # Trainer Class
    trainer = 0
    if (load_from_checkpoint == False):
        trainer = AutoSteerTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    else:
        trainer = AutoSteerTrainer(checkpoint_path=checkpoint_path, is_pretrained=True)

    trainer.zero_grad()

    # Total training epochs
    num_epochs = 50
    batch_size = 5

    # Epochs
    for epoch in range(0, num_epochs):

        # Printing epochs
        print('Epoch: ', epoch + 1)

        # Randomizing data
        # randomlist_train_data = random.sample(range(1, num_train_samples), total_train_samples)
        # randomlist_train_data = range(1, total_train_samples)

        # Batch size schedule
        if (epoch >= 5 and epoch < 10):
            batch_size = 4

        if (epoch >= 10):
            batch_size = 3

        # Learning rate schedule
        if (epoch >= 2 and epoch < 15):
            trainer.set_learning_rate(0.0001)
        if (epoch >= 15):
            trainer.set_learning_rate(0.000025)

        # Augmentations schedule
        apply_augmentations = False
        # if (epoch >= 10 and epoch < 25):
        #     apply_augmentations = True
        # if (epoch >= 25):
        #     apply_augmentations = False

        # Loop through data
        for count in range(0, total_train_samples):

            # Log counter
            train_log_count = count + total_train_samples * epoch

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            # Get data
            # image, gt = auto_steer_dataset.getItem(randomlist_train_data[count])
            img_T_minus_1, gt_T_minus_1, img_T, gt_T = auto_steer_dataset.train[count]

            # Assign Data
            trainer.set_data(img_T_minus_1, gt_T_minus_1, img_T, gt_T)

            #     # ### Train on flipped images ### #
            if random.random() < 0.5:
                trainer.apply_augmentations()

            # Converting to tensor and loading
            trainer.load_data()

            # Run model and calculate loss
            trainer.run_model()

            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if ((count + 1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if ((count + 1) % 250 == 0):
                trainer.log_loss(train_log_count)

            # Logging Image to Tensor Board every 1000 steps
            if ((count + 1) % 1000 == 0):
                trainer.save_visualization(train_log_count)

        # Save model and run validation on entire validation
        # dataset after each epoch

        # Save Model
        model_save_path = model_save_root_path + 'AutoSteer_iter_' + \
                          str(count + total_train_samples * epoch) \
                          + '_epoch_' + str(epoch) + '_step_' + \
                          str(count) + '.pth'

        trainer.save_model(model_save_path)

        # Test and save visualization
        # print('Testing')
        # trainer.test(test_images, test_images_save_path, train_log_count)

        # # Validate
        print('Validating')

        # Setting model to evaluation mode
        trainer.set_eval_mode()

        # Overall IoU
        running_cel_loss = 0
        running_l1_loss = 0
        running_val_accuracy = 0

        # No gradient calculation
        with torch.no_grad():

            # AutoSteer
            for val_count in range(0, total_val_samples):
                # image_val, gt_val = auto_steer_dataset.getItemVal(val_count)
                # frame_id, image_val, gt_val = auto_steer_dataset.getItem(val_count)
                img_T_minus_1_val, gt_T_minus_1_val, img_T_val, gt_T_val = auto_steer_dataset.val[val_count]

                # Run Validation and calculate IoU Score
                cel_loss, accuracy = trainer.validate(img_T_minus_1_val, gt_T_minus_1_val, img_T_val, gt_T_val)

                # Accumulate individual IoU scores for validation samples
                running_cel_loss += cel_loss
                running_val_accuracy += accuracy

            # Calculating average loss of complete validation set
            val_cel_loss = running_cel_loss / total_val_samples
            val_accuracy = running_val_accuracy / total_val_samples

            # Logging average validation loss to TensorBoard
            trainer.log_val(val_cel_loss, val_accuracy, epoch)

        # Resetting model back to training
        trainer.set_train_mode()

    trainer.cleanup()


if __name__ == '__main__':
    main()
# %%
