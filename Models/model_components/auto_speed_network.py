import torch
import sys
sys.path.append('../../')
from Models.model_components.auto_speed_backbone import AutoSpeedBackbone
from Models.model_components.auto_speed_neck import AutoSpeedNeck
from Models.model_components.auto_speed_head import AutoSpeedHead
from Models.model_components.auto_speed_layers import Conv


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = AutoSpeedBackbone(width, depth, csp)
        self.fpn = AutoSpeedNeck(width, depth, csp)

        # img_dummy = torch.zeros(1, width[0], 256, 256)
        img_dummy = torch.zeros(1, width[0], 640, 640)
        self.head = AutoSpeedHead(num_classes, (width[3], width[4], width[5]))
        # self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.head.stride = torch.tensor([640 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class AutoSpeedNetwork:
    def __init__(self):
        self.dynamic_weighting = {
            'n': {'csp': [False, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 16, 32, 64, 128, 256]},
            's': {'csp': [False, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 32, 64, 128, 256, 512]},
            'm': {'csp': [True, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 64, 128, 256, 512, 512]},
            'l': {'csp': [True, True], 'depth': [2, 2, 2, 2, 2, 2], 'width': [3, 64, 128, 256, 512, 512]},
            'x': {'csp': [True, True], 'depth': [2, 2, 2, 2, 2, 2], 'width': [3, 96, 192, 384, 768, 768]},
        }

    def build_model(self, version, num_classes):
        csp = self.dynamic_weighting[version]['csp']
        depth = self.dynamic_weighting[version]['depth']
        width = self.dynamic_weighting[version]['width']
        return YOLO(width, depth, csp, num_classes)
