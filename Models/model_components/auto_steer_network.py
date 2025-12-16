#! /usr/bin/env python3
import torch
import torch.nn as nn

class AutoSteerNetwork(nn.Module):
    def __init__(self):
        super(AutoSteerNetwork, self).__init__()

        # Standard
        self.GeLU = nn.GELU()
        self.pool = nn.MaxPool2d(2, stride=2)

        # Lane Mask - Decode Layers
        self.decode_layer_0 = nn.Conv2d(6, 32, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.decode_layer_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(32, 32, 3, 1, 1)

        self.dropout_aggressize = nn.Dropout(p=0.4)

        # Steering Angle - Prediction Layers
        self.steering_pred_layer_prev_0 = nn.Linear(1600, 1600)
        self.steering_pred_layer_prev_1 = nn.Linear(1600, 61)

        # Steering Angle - Prediction Layers
        self.steering_pred_layer_0 = nn.Linear(1600, 1600)
        self.steering_pred_layer_1 = nn.Linear(1600, 61)

    def forward(self, lane_features_concat):

        # 80 by 160 by 6
        s0 = self.decode_layer_0(lane_features_concat)
        s0 = self.GeLU(s0)
        s1 = self.pool(s0)

        # Creating skip connection for s1 feature
        skip_1 = self.pool(s1)
        skip_1 = self.pool(skip_1)
        skip_1 = self.pool(skip_1)

        # 40 by 80 by 32
        s1 = self.decode_layer_1(s1)
        s1 = self.GeLU(s1)
        s2 = self.pool(s1)

        # Creating skip connection for s2 feature
        skip_2 = self.pool(s2)
        skip_2 = self.pool(skip_2)

        # 20 by 40 by 32
        s2 = self.decode_layer_2(s2)
        s2 = self.GeLU(s2)
        s3 = self.pool(s2)

        # Creating skip connection for s3 feature
        skip_3 = self.pool(s3)

        # 10 by 20 by 32
        s3 = self.decode_layer_3(s3)
        s3 = self.GeLU(s3)
        s4 = self.pool(s3)

        # Low level features 5 by 10 by 32
        steering_angle_features = s4 + skip_3 + skip_2 + skip_1


        # Create feature vector - 1600
        feature_vector = torch.flatten(steering_angle_features)

        steering_angle_prev = self.steering_pred_layer_prev_0(feature_vector)
        steering_angle_prev = self.GeLU(steering_angle_prev)
        steering_angle_prev = self.dropout_aggressize(steering_angle_prev)
        steering_angle_prediction_prev = self.steering_pred_layer_prev_1(steering_angle_prev)

        steering_angle = self.steering_pred_layer_0(feature_vector)
        steering_angle = self.GeLU(steering_angle)
        steering_angle = self.dropout_aggressize(steering_angle)
        steering_angle_prediction = self.steering_pred_layer_1(steering_angle)

        # A vector of length 61 where each position encodes a steering angle
        # -30, -29, -28, -27.....0......27, 28, 29, 30
        # Trained as a classificaiton problem, where the argmax indicates the steering angle
        # Cross Entropy Loss
        
        return steering_angle_prediction_prev, steering_angle_prediction