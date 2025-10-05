import torch
from torchvision import transforms
import sys
sys.path.append("..")
from model_components.auto_steer_network import AutoSteerNetwork


class AutoSteerNetworkInfer():
    def __init__(
            self, 
            checkpoint_path: str = ""
    ):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], 
                    std  = [0.229, 0.224, 0.225]
                )
            ]
        )
            
        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {self.device} for inference")
            
        # Init model
        self.model = AutoSteerNetwork()
        if (checkpoint_path):
            print(f"Loading trained AutoSteer checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(
                    checkpoint_path, 
                    weights_only = True
                )
            )
        else:
            print("Loading vanilla AutoSteer model for training")
        
        # Load model to device, set eval mode
        self.model.to(self.device)
        self.model.eval()

    
    def inference(self, image):

        # Load image
        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)
    
        # Run model
        prediction = self.model(image_tensor)

        # Get output
        binary_segggggg, path_prediction = prediction
        binary_segggggg = binary_segggggg.squeeze(0).cpu().detach().numpy()
        path_prediction = path_prediction.squeeze(0).cpu().detach().numpy()

        return binary_segggggg, path_prediction