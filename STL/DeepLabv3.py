import torch
import torch.nn as nn
import torch.nn.functional as F
from ASPP import ASPP_Bottleneck
from pl_bolts.models.self_supervised.resnets import resnet50

feature_extracting = True

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)  # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)  # (*blocks: call with unpacked list entires as arguments)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(
            x)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(
            out))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(
            x)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(
            out)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x)))  # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(
            out)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(
            out))  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(
            x)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(
            out)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out

# U-Net Decoder 모듈 정의

# DeepLabv3+ 모델 정의
class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=3): # seg = 3, det = 2
        super(DeepLabv3Plus, self).__init__()

        resnet = resnet50(pretrained=False)
        MODEL_PATH = "/home/skim/swav_rn50_ep200.torch"
        state_dict = torch.load(MODEL_PATH, map_location='cuda:0')
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('encoder.', '')] = state_dict.pop(key)

            resnet = load_encoder_weights(resnet, state_dict)
        set_parameter_requires_grad(resnet, feature_extracting)

        # ASPP (only segmentation)
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

        # Low-level feature projection
        self.low_level_conv = nn.Conv2d(512, 48, kernel_size=1)
        self.low_level_bn = nn.BatchNorm2d(48)

        # Decoder
        self.decoder_conv1_seg = nn.Conv2d(num_classes + 48, 256, kernel_size=3, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(256)

        # Final segmentation output
        self.final_conv_seg = nn.Conv2d(256, num_classes_seg, kernel_size=1)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_encoder_weights(encoder, weights):
    model_dict = encoder.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    encoder.load_state_dict(model_dict)

    return encoder
