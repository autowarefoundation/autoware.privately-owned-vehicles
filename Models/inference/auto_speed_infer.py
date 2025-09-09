import torch
from torchvision import transforms
from torchvision import ops


class AutoSpeedNetworkInfer():
    def __init__(self, checkpoint_path=''):
        # Image loader
        self.train_size = (640, 640)
        self.image_loader = transforms.Compose([
            transforms.Resize(self.train_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float16),
        ])

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        # Instantiate model, load to device and set to evaluation mode
        self.model = torch.load(checkpoint_path + "/best.pt", map_location="cpu", weights_only=False)['model']
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def xywh2xyxy(self, x):
        # Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    def nms(self, preds, iou_thres=0.45):
        if preds.numel() == 0:
            return torch.empty(0, 6)

        # Separate bounding boxes, scores, and class indices
        boxes = self.xywh2xyxy(preds[:, :4])
        scores = preds[:, 4]
        classes = preds[:, 5]

        # Apply torchvision's NMS
        keep_indices = ops.nms(boxes, scores, iou_thres)

        # Return the detections that were kept
        kept_preds = preds[keep_indices]

        return kept_preds

    def post_process_predictions(self, raw_preds, conf_thres, iou_thres):
        # Reshape and permute the tensor
        preds = raw_preds.squeeze(0).permute(1, 0)

        # Separate bounding boxes and class scores
        boxes = preds[:, :4]
        class_probs = preds[:, 4:]

        # Get the best score and class index for each prediction
        scores, class_ids = torch.max(class_probs.sigmoid(), dim=1)

        # Apply confidence threshold
        valid_preds_mask = scores > conf_thres
        boxes = boxes[valid_preds_mask]
        scores = scores[valid_preds_mask]
        class_ids = class_ids[valid_preds_mask]

        if boxes.numel() == 0:
            return torch.empty(0, 6)

        # Combine into a single tensor for NMS
        combined_preds = torch.cat([boxes, scores.unsqueeze(1), class_ids.unsqueeze(1).float()], dim=1)

        # Apply your separate NMS function
        final_preds = self.nms(combined_preds, iou_thres)

        return final_preds

    def inference(self, image):
        orig_w, orig_h = image.size

        image_tensor = self.image_loader(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Run model
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Post-processing
        final_prediction = self.post_process_predictions(prediction, conf_thres=0.60, iou_thres=0.45)

        if final_prediction.numel() == 0:
            return []

        # Rescale boxes to original image size
        scale_w = orig_w / self.train_size[0] / 2
        scale_h = orig_h / self.train_size[1] / 2

        final_prediction[:, [0, 2]] *= scale_w
        final_prediction[:, [1, 3]] *= scale_h

        return final_prediction.tolist()
