import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroThresholdConstraint:
    def __init__(self, threshold=1e-2):
        self.threshold = threshold
    
    def __call__(self, w):
        mask = torch.abs(w) >= self.threshold
        return w * mask.float()


class ConstrainedConv2d(nn.Conv2d):
    """Conv2d with weight constraints - inherits directly from nn.Conv2d"""
    def __init__(self, *args, constraint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint
    
    def forward(self, x):
        if self.constraint is not None:
            with torch.no_grad():
                self.weight.data = self.constraint(self.weight.data)
        return super().forward(x)


class FMRINet(nn.Module):
    """Simplified FMRINet using standard PyTorch CHW format"""
    def __init__(self, temporal_filters=8, num_classes=6, input_shape=(1, 214, 277),
                 depth_multiplier=4, zero_thresh=1e-2, dropout_in=0.25,
                 dropout_mid=0.5, name="fmriNet", debug=False):
        super().__init__()
        self.name = name
        self.input_shape = input_shape  # PyTorch style (C, H, W)
        self.debug = debug

        C, H, W = input_shape
        assert C == 1, "This implementation expects single-channel input"

        # Input Dropout
        self.dropout_in = nn.Dropout(dropout_in)

        # Conv2D with SAME padding (1x60)
        pad_w = 60 // 2  # SAME padding for kernel_size=60
        self.conv1 = nn.Conv2d(C, temporal_filters, (1, 60), 
                              padding=(0, pad_w), bias=False)

        # DepthwiseConv2D (HxW -> 1xW) with zero threshold constraint
        constraint = ZeroThresholdConstraint(threshold=zero_thresh)
        self.depthwise_conv = ConstrainedConv2d(
            in_channels=temporal_filters,
            out_channels=temporal_filters * depth_multiplier,
            kernel_size=(H, 1),  # (214, 1) - spatial filter
            groups=temporal_filters,
            bias=False,
            constraint=constraint
        )
        self.bn1 = nn.BatchNorm2d(temporal_filters * depth_multiplier)

        # AvgPool (1x15)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.dropout_mid1 = nn.Dropout(dropout_mid)

        # Separable Conv2D SAME (1x8)
        pad_w2 = 8 // 2  # SAME padding
        self.separable_depthwise = nn.Conv2d(
            temporal_filters * depth_multiplier,
            temporal_filters * depth_multiplier,
            kernel_size=(1, 8),
            padding=(0, pad_w2),
            groups=temporal_filters * depth_multiplier,
            bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            temporal_filters * depth_multiplier,
            64,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)

        # AvgPool (1x4)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout_mid2 = nn.Dropout(dropout_mid)

        # Calculate linear input size
        self._calculate_linear_input_size()
        if self.debug:
            print(f"DEBUG: Calculated linear input size: {self.linear_input_size}")
        
        # Dense classifier
        self.classifier = nn.Linear(self.linear_input_size, num_classes)

    def _log(self, *args):
        if self.debug: 
            print(*args)

    def _calculate_linear_input_size(self):
        """Calculate the size after all conv/pool operations"""
        with torch.no_grad():
            C, H, W = self.input_shape
            dummy = torch.zeros(1, C, H, W)  # CHW format
            x = self._forward_features(dummy)
            self.linear_input_size = x.numel() // x.size(0)

    def _forward_features(self, x):
        """Forward pass through feature extraction layers"""
        self._log("Input:", x.shape)

        # Input dropout
        x = self.dropout_in(x)

        # Conv1: (C,H,W) -> (temporal_filters,H,W)
        x = self.conv1(x)
        self._log("After conv1:", x.shape)

        # Spatial depthwise conv with constraint: (temporal_filters,H,W) -> (temporal_filters*4,1,W)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        self._log("After depthwise:", x.shape)

        # AvgPool1 + Dropout
        x = self.avgpool1(x)
        self._log("After avgpool1:", x.shape)
        x = self.dropout_mid1(x)

        # Separable Conv2D
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        self._log("After separable:", x.shape)

        # AvgPool2 + Dropout
        x = self.avgpool2(x)
        self._log("After avgpool2:", x.shape)
        x = self.dropout_mid2(x)

        return x

    def forward(self, x):
        """
        Forward pass
        Input: x should be in CHW format (B, C, H, W) where C=1, H=214, W=277
        """
        C, H, W = self.input_shape
        
        # Handle single sample input
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Validate input shape
        assert x.shape[1:] == (C, H, W), f"Expected input shape (B,{C},{H},{W}), got {x.shape}"
        
        # Extract features
        x = self._forward_features(x)
        
        # Flatten and classify
        x = x.flatten(1)  # Flatten all dims except batch
        logits = self.classifier(x)
        
        return logits


# Helper functions
def build_fmri_net(temporal_filters=8, num_classes=6, input_shape=(1, 214, 277),
                   depth_multiplier=4, zero_thresh=1e-2, dropout_in=0.25,
                   dropout_mid=0.5, name="fmriNet", use_cuda=True, debug=False):
    """Build fMRINet model"""
    model = FMRINet(
        temporal_filters=temporal_filters,
        num_classes=num_classes,
        input_shape=input_shape,
        depth_multiplier=depth_multiplier,
        zero_thresh=zero_thresh,
        dropout_in=dropout_in,
        dropout_mid=dropout_mid,
        name=name,
        debug=debug
    )
    
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    
    return model


def fmriNet8(num_classes=6, input_shape=(1, 214, 277), use_cuda=True, **kwargs):
    return build_fmri_net(8, num_classes, input_shape, name="fmriNet8", use_cuda=use_cuda, **kwargs)

def fmriNet16(num_classes=6, input_shape=(1, 214, 277), use_cuda=True, **kwargs):
    return build_fmri_net(16, num_classes, input_shape, name="fmriNet16", use_cuda=use_cuda, **kwargs)

def fmriNet32(num_classes=6, input_shape=(1, 214, 277), use_cuda=True, **kwargs):
    return build_fmri_net(32, num_classes, input_shape, name="fmriNet32", use_cuda=use_cuda, **kwargs)