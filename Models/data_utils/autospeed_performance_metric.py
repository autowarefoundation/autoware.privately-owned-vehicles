import os
from tqdm import tqdm
import json
from argparse import ArgumentParser
import cv2
from PIL import Image
from Models.inference.auto_speed_infer import AutoSpeedNetworkInfer
from torchvision.ops import box_iou
import torch

predictions_color_map = {
    1: (0, 0, 100),  # red
    2: (0, 100, 100),  # yellow
    3: (100, 100, 0)  # cyan
}

labels_color_map = {
    1: (0, 0, 255),  # red
    2: (0, 255, 255),  # yellow
    3: (255, 255, 0)  # cyan
}


def make_visualization(labels, predictions, input_image_filepath):
    img_cv = cv2.imread(input_image_filepath)

    for label in labels:
        cls, x, y, width, height = label

        x1 = float(x)
        x2 = float(x) + float(width)
        y1 = float(y)
        y2 = float(y) + float(height)

        # Pick color, fallback to white if unknown class
        color = labels_color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred

        # Pick color, fallback to white if unknown class
        color = predictions_color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        # label = f"Class: {int(cls)} | Score: {conf:.2f}"
        # cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Prediction Objects', img_cv)
    cv2.waitKey(0)


def make_visualization_predictions(predictions, input_image_filepath):
    img_cv = cv2.imread(input_image_filepath)
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred

        # Pick color, fallback to white if unknown class
        color = labels_color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        # label = f"Class: {int(cls)} | Score: {conf:.2f}"
        # cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Prediction Objects', img_cv)
    cv2.waitKey(0)


def make_visualization_labels(labels, input_image_filepath):
    img_cv = cv2.imread(input_image_filepath)
    for label in labels:
        cls, x, y, width, height = label

        x1 = float(x)
        x2 = float(x) + float(width)
        y1 = float(y)
        y2 = float(y) + float(height)

        # Pick color, fallback to white if unknown class
        color = labels_color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    cv2.imshow('Prediction Objects', img_cv)
    cv2.waitKey(0)


def get_labels(file):
    labels = []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for box in data["result"]:
            id = box["id"] if "id" in box else box["attribute"]
            labels.append([id, box["x"], box["y"], box["width"], box["height"]])
    return labels


def get_label_boxes(labels):
    label_boxes = []
    for label in labels:
        cls, x, y, width, height = label
        if cls == 1:
            x1 = float(x)
            x2 = float(x) + float(width)
            y1 = float(y)
            y2 = float(y) + float(height)
            label_boxes.append([x1, y1, x2, y2])
    return label_boxes


def get_prediction_boxes(predictions, clzz=None):
    prediction_boxes = []
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        if clzz is not None:
            if int(cls) == 1:
                prediction_boxes.append([x1, y1, x2, y2])
        else:
            prediction_boxes.append([x1, y1, x2, y2])

    return prediction_boxes


def prediction_performance(labels, predictions, iou_threshold=0.75):
    labels = torch.tensor(labels, dtype=torch.float)
    predictions = torch.tensor(predictions, dtype=torch.float)
    """
        labels: Tensor [N, 4] (x1, y1, x2, y2)
        predictions: Tensor [M, 4] (x1, y1, x2, y2)
        """
    if len(labels) == 0:
        return 0.0 if len(predictions) > 0 else 1.0

    if len(predictions) == 0:
        return 0.0  # no predictions, all IoUs = 0

    iou_matrix = box_iou(predictions, labels)  # [M, N]

    matched_gt = set()
    ious = []

    for pred_idx in range(len(predictions)):
        # best GT for this prediction
        gt_idx = torch.argmax(iou_matrix[pred_idx]).item()
        iou_val = iou_matrix[pred_idx, gt_idx].item()

        if iou_val >= iou_threshold and gt_idx not in matched_gt:
            ious.append(iou_val)
            matched_gt.add(gt_idx)

    # Add 0 IoU for unmatched GT boxes
    for gt_idx in range(len(labels)):
        if gt_idx not in matched_gt:
            ious.append(0.0)

    return sum(ious) / len(labels)


def get_gt_cipo_label(labels):
    cipo_label = None

    for label in labels:
        cls = label[0]
        if cls == 1:  # CIPO
            cipo_label = label
    return cipo_label


def get_predict_cipo_label(predictions):
    cipo_label = None

    for prediction in predictions:
        cls = prediction[5]
        if cls == 1:  # CIPO
            cipo_label = prediction
    return cipo_label


def calculate_max_iou(labels, predictions):
    max_iou = 1.0
    cipo_label = get_gt_cipo_label(labels)
    if cipo_label is not None:
        label_boxes = get_label_boxes([cipo_label])
        prediction_boxes = get_prediction_boxes(predictions)

        """
        label_boxes_tensor: Tensor [1, 4] (x1, y1, x2, y2)
        prediction_boxes_tensor: Tensor [M, 4] (x1, y1, x2, y2)
        """
        label_boxes_tensor = torch.tensor(label_boxes, dtype=torch.float)
        prediction_boxes_tensor = torch.tensor(prediction_boxes, dtype=torch.float)

        if len(label_boxes_tensor) == 0:
            return 0.0 if len(prediction_boxes_tensor) > 0 else 1.0

        if len(prediction_boxes_tensor) == 0:
            return 0.0  # no predictions, all IoUs = 0

        iou_matrix = box_iou(label_boxes_tensor, prediction_boxes_tensor)  # [1, M]

        pred_idx = torch.argmax(iou_matrix[0]).item()
        max_iou = iou_matrix[0, pred_idx].item()
        # print("max_iou: " + str(max_iou))
    # else:
    #     predict_cipo_label = get_predict_cipo_label(predictions)
    #     if predict_cipo_label:
    #         print("//////////////////////////////")
    #     max_iou = 0.0 if predict_cipo_label else 1.0
    #     # print("max_iou: " + str(max_iou))

    return max_iou


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-d", "--dataset", help="dataset directory path")
    parser.add_argument("-s", "--sequence", help="sequence name")

    # parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath",
    #                     help="path to input image which will be processed by DomainSeg")
    args = parser.parse_args()
    model_checkpoint_path = args.model_checkpoint_path

    model = AutoSpeedNetworkInfer(model_checkpoint_path)

    images_dir = os.path.join(args.dataset, "images/val", args.sequence)
    # images_dir = os.path.join(args.dataset, "images/train", args.sequence)
    labels_dir = os.path.join(args.dataset, "labels/val", args.sequence)
    # labels_dir = os.path.join(args.dataset, "labels/train", args.sequence)

    for threshold in [0.50, 0.75, 0.90]:
        c = 0
        image_num = len(os.listdir(images_dir))
        for image in tqdm(os.listdir(images_dir), desc="Processing sequence"):
            image_path = os.path.join(images_dir, image)
            labels = get_labels(os.path.join(labels_dir, image) + ".json")
            predictions = model.inference(Image.open(image_path))
            # make_visualization_labels(labels, image_path)
            # make_visualization_predictions(prediction, image_path)
            # make_visualization(labels, predictions, image_path)
            # s += prediction_performance(get_label_boxes(labels), get_prediction_boxes(predictions))
            max_iou = calculate_max_iou(labels, predictions)

            if max_iou > threshold:
                c += 1
        # print(c / image_num)
        # mIoU = s / image_num
        # print(f"mIoU: {format(mIoU * 100, '.2f')}%")
        print(f"threshold: {threshold}  maxIoU: {format((c / image_num) * 100, '.2f')}%")
