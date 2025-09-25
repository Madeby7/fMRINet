import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    """Conv2d that works directly with HWC format (B,H,W,C)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
    
    def forward(self, x):
        # x: (B,H,W,C) -> (B,C,H,W) -> Conv -> (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # CHW -> HWC
        return x


class BatchNorm2d(nn.Module):
    """BatchNorm2d that works directly with HWC format (B,H,W,C)"""
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
    
    def forward(self, x):
        # x: (B,H,W,C) -> (B,C,H,W) -> BN -> (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)  # CHW -> HWC
        return x


class AvgPool2d(nn.Module):
    """AvgPool2d that works directly with HWC format (B,H,W,C)"""
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, **kwargs)
    
    def forward(self, x):
        # x: (B,H,W,C) -> (B,C,H,W) -> Pool -> (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1)  # CHW -> HWC
        return x


class TFSamePad2d(nn.Module):
    """TF-like SAME padding for HWC format"""
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__()
        if isinstance(kernel_size, int):   kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):        stride = (stride, stride)
        if isinstance(dilation, int):      dilation = (dilation, dilation)
        self.kh, self.kw = kernel_size
        self.sh, self.sw = stride
        self.dh, self.dw = dilation

    def forward(self, x):
        # x: (B,H,W,C)
        ih, iw = x.shape[1], x.shape[2]  # H, W dimensions
        oh = math.ceil(ih / self.sh)
        ow = math.ceil(iw / self.sw)
        pad_h = max((oh - 1) * self.sh + (self.kh - 1) * self.dh + 1 - ih, 0)
        pad_w = max((ow - 1) * self.sw + (self.kw - 1) * self.dw + 1 - iw, 0)
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        # Pad in (H,W) dimensions: pad order is (left, right, top, bottom)
        return F.pad(x, (0, 0, pad_left, pad_right, pad_top, pad_bottom))


class ZeroThresholdConstraint:
    def __init__(self, threshold=2e-2):
        self.threshold = threshold
    
    def __call__(self, w):
        # Return the modified tensor
        mask = torch.abs(w) >= self.threshold
        return w * mask.float()


class ConstrainedConv2d(Conv2d):
    """Conv2d with weight constraints"""
    def __init__(self, *args, constraint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint
    
    def forward(self, x):
        if self.constraint is not None:
            with torch.no_grad():
                self.conv.weight.data = self.constraint(self.conv.weight.data)
        return super().forward(x)


class SeparableConv2d(nn.Module):
    """Separable Conv2d for HWC format"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.samepad = TFSamePad2d(kernel_size)
        self.depthwise = Conv2d(in_channels, in_channels, kernel_size,
                               padding=0, groups=in_channels, bias=False)
        self.pointwise = Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        x = self.samepad(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FMRINet(nn.Module):
    """Clean HWC-native FMRINet implementation"""
    def __init__(self, temporal_filters=8, num_classes=6, input_shape=(214, 277, 1),
                 depth_multiplier=4, zero_thresh=1e-4, dropout_in=0.25,
                 dropout_mid=0.5, name="fmriNet", debug=False):
        super().__init__()
        self.name = name
        self.input_shape = input_shape  # TF style (H, W, C)
        self.debug = debug

        H, W, C = input_shape
        assert C == 1, "This implementation expects single-channel input to match the TF model."

        # Input Dropout
        self.dropout_in = nn.Dropout(dropout_in)

        # Conv2D SAME (1x60), no bias
        self.samepad1 = TFSamePad2d((1, 60))
        self.conv1 = Conv2d(C, temporal_filters, (1, 60), padding=0, bias=False)

        # DepthwiseConv2D VALID (1xH), depth_multiplier=4, no bias, with constraint
        constraint = ZeroThresholdConstraint(threshold=zero_thresh)
        self.depthwise_conv = ConstrainedConv2d(
            in_channels=temporal_filters,
            out_channels=temporal_filters * depth_multiplier,
            kernel_size=(1, H),  # (1, 214)
            groups=temporal_filters,
            bias=False,
            constraint=constraint,
            padding=0
        )
        self.bn1 = BatchNorm2d(temporal_filters * depth_multiplier)

        # AvgPool (1x15) VALID + Dropout
        self.avgpool1 = AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.dropout_mid1 = nn.Dropout(dropout_mid)

        # SeparableConv2D SAME (1x8), no bias
        self.separable_conv = SeparableConv2d(
            in_channels=temporal_filters * depth_multiplier,
            out_channels=64,
            kernel_size=(1, 8)
        )
        self.bn2 = BatchNorm2d(64)

        # AvgPool (1x4) VALID + Dropout
        self.avgpool2 = AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout_mid2 = nn.Dropout(dropout_mid)

        # Dense head
        self._calculate_linear_input_size()
        if self.debug:
            print(f"DEBUG: Calculated linear input size: {self.linear_input_size}")
        self.dense = nn.Linear(self.linear_input_size, num_classes)

    def _log(self, *a):
        if self.debug: print(*a)

    def _calculate_linear_input_size(self):
        with torch.no_grad():
            H, W, C = self.input_shape
            dummy = torch.zeros(1, H, W, C)  # HWC format
            x = self._forward_features(dummy)
            self.linear_input_size = x.numel() // x.size(0)

    def _forward_features(self, x):
        # x: (B,H,W,C) - channels last format (TensorFlow-like)
        self._log("input HWC:", x.shape)

        # Input dropout
        x = self.dropout_in(x)

        # Conv1 SAME (1x60)
        x = self.samepad1(x)
        x = self.conv1(x)  # -> (B,H,W,temporal_filters)
        self._log("after conv1 HWC:", x.shape)

        # Permute dimensions for TF-like behavior: (H,W,C) -> (W,H,C)
        x = x.permute(0, 2, 1, 3)  # -> (B,W,H,temporal_filters)
        self._log("after permute HWC:", x.shape)

        # Depthwise VALID (1xH=214), mult=4
        x = self.depthwise_conv(x)  # -> (B,W,1,temporal_filters*4)
        x = self.bn1(x)
        x = F.relu(x)
        self._log("after depthwise HWC:", x.shape)

        # Permute back: (W,H,C) -> (H,W,C)
        x = x.permute(0, 2, 1, 3)  # -> (B,1,W,temporal_filters*4)
        self._log("after permute back HWC:", x.shape)

        # AvgPool (1x15) VALID + Dropout
        x = self.avgpool1(x)  # -> (B,1,W_pooled,temporal_filters*4)
        self._log("after avgpool1 HWC:", x.shape)
        x = self.dropout_mid1(x)

        # SeparableConv2D SAME (1x8)
        x = self.separable_conv(x)  # -> (B,1,W_sep,64)
        x = self.bn2(x)
        x = F.relu(x)
        self._log("after separable HWC:", x.shape)

        # AvgPool (1x4) VALID + Dropout
        x = self.avgpool2(x)  # -> (B,1,W_final,64)
        self._log("after avgpool2 HWC:", x.shape)
        x = self.dropout_mid2(x)

        return x

    def forward(self, x):
        # Input should be in HWC format: (B,H,W,C)
        H, W, C = self.input_shape
        if x.dim() == 3:  # If single sample
            x = x.unsqueeze(0)
        
        assert x.shape[1:] == (H, W, C), f"Expected input shape (B,{H},{W},{C}), got {x.shape}"
        
        x = self._forward_features(x)  # Stays in HWC format throughout
        x = x.reshape(x.size(0), -1)   # Flatten for dense layer
        logits = self.dense(x)         # -> (B,num_classes)
        return logits


# Helper functions
def build_fmri_net(temporal_filters=8, num_classes=6, input_shape=(214, 277, 1),
                   depth_multiplier=4, zero_thresh=1e-2, dropout_in=0.25,
                   dropout_mid=0.5, name="fmriNet", use_cuda=True, debug=False):
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


def fmriNet8(num_classes=6, input_shape=(214, 277, 1), use_cuda=True, **kwargs):
    return build_fmri_net(8, num_classes, input_shape, name="fmriNet8", use_cuda=use_cuda, **kwargs)

def fmriNet16(num_classes=6, input_shape=(214, 277, 1), use_cuda=True, **kwargs):
    return build_fmri_net(16, num_classes, input_shape, name="fmriNet16", use_cuda=use_cuda, **kwargs)

def fmriNet32(num_classes=6, input_shape=(214, 277, 1), use_cuda=True, **kwargs):
    return build_fmri_net(32, num_classes, input_shape, name="fmriNet32", use_cuda=use_cuda, **kwargs)