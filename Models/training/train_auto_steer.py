# Main train loop for AutoSteer 
# Responsible for getting data via the AutoSteer data loader, passing the data to the AutoSteer trainer class
# Running the main train loop over multiple epochs
# Does simulation of batch size (we use a batch size of one)
# Saving the model checkpoints
# Colating the validation metrics