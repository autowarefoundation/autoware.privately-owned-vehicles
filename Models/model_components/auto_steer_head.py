#! /usr/bin/env python3
import torch
import torch.nn as nn

class AutoSteerHead(nn.Module):
    def __init__(self):
        super(AutoSteerHead, self).__init__()
        
        # Standard
        self.GeLU = nn.GELU()
        self.dropout_aggressize = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)
        
        # Neck Reduction Layer
        self.neck_reduce_layer_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.neck_reduce_layer_2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.neck_reduce_layer_3 = nn.Conv2d(64, 64, 3, 1, 1)

        # Road shape decoding layers
        self.decode_layer_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.decode_layer_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(64, 1, 3, 1, 1)
        
        # Steering angle decoding layers
        self.steering_decode_layer = nn.Linear(800, 800)
        self.steering_output = nn.Linear(800, 1)



    def forward(self, context, neck, feature_prev, feature_prev_prev):

        # Calculating feature vector

        # Reducing size of neck to match context
        p0 = self.pool(neck)
        p0 = self.pool(p0)

        # Pseudo-attention
        p0 = p0*context + context

        # Reduction
        p1 = self.neck_reduce_layer_1(p0)
        p1 = self.GeLU(p1)
        p2 = self.neck_reduce_layer_2(p1)
        p2 = self.GeLU(p2)
        p3 = self.neck_reduce_layer_3(p2)
        feature = self.GeLU(p3)


        # Extract Spatio-Temporal Path Information
        spatiotemporal_features = torch.cat((feature, feature_prev, feature_prev_prev), 3)
        spatiotemporal_features = self.decode_layer_1(spatiotemporal_features)
        spatiotemporal_features = self.GeLU(spatiotemporal_features)
        spatiotemporal_features = self.decode_layer_2(spatiotemporal_features)
        spatiotemporal_features = self.GeLU(spatiotemporal_features)
        spatiotemporal_features = self.decode_layer_3(spatiotemporal_features)
        spatiotemporal_features = self.GeLU(spatiotemporal_features)
        
        # Create feature vector
        feature_vector = torch.flatten(p3)

        # Extract Spatio-Temporal Path Information
        steering_angle = self.steering_decode_layer(feature_vector)
        steering_angle = self.GeLU(steering_angle)
        steering_angle = self.steering_output(steering_angle)

        return steering_angle, feature