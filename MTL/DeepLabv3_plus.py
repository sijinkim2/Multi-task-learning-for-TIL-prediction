import torch
import torch.nn as nn
import torch.nn.functional as F
from ASPP import ASPP, ASPP_Bottleneck
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


# DeepLabv3+ 모델 정의
class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes_seg=3, num_classes_det=2):
        super(DeepLabv3Plus, self).__init__()

        resnet = resnet50(pretrained=False)
        MODEL_PATH = "/home/skim/swav_rn50_ep200.torch"
        state_dict = torch.load(MODEL_PATH, map_location='cuda:0')
        for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '').replace('encoder.', '')] = state_dict.pop(key)

                resnet = load_encoder_weights(resnet, state_dict)
        set_parameter_requires_grad(resnet, feature_extracting)

        self.conv1 = nn.Sequential(*list(resnet.children())[:1])  # Conv1 Layer (7x7)
        self.bn1 = nn.Sequential(*list(resnet.children())[1:2])  # BatchNorm Layer
        self.relu = nn.Sequential(*list(resnet.children())[2:3])  # ReLU Activation
        self.maxpool = nn.Sequential(*list(resnet.children())[3:4])  # MaxPool Layer

        # Residual Blocks
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5])  # Layer1 (256 채널 출력)
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # Layer2 (512 채널 출력)
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # Layer3 (1024 채널 출력)
        self.layer4 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)


        self.aspp = ASPP_Bottleneck(num_class_seg=3, num_class_det=2)  # ASPP 모듈 사용, detection

        # c2 layer에 적용
        self.low_level_conv = nn.Conv2d(512, 48, kernel_size=1)
        self.low_level_bn = nn.BatchNorm2d(48)

        # Decoder
        self.decoder_conv1 = nn.Conv2d(num_classes_seg + num_classes_det + 48, 256, kernel_size=3, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(256)

        # 최종 컨볼루션
        self.final_conv_seg = nn.Conv2d(256, 3, kernel_size=1)
        self.final_conv_det = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # ResNet Backbone
        c = self.conv1(x)  # Conv1 통과
        b = self.bn1(c)  # BatchNorm 통과
        r = self.relu(b)  # ReLU 활성화 통과
        m = self.maxpool(r)  # MaxPooling 통과

        # Residual Block 통과 (Layer1 ~ Layer3)
        c1 = self.layer1(m)  # (batch_size, 256, h/4, w/4)
        c2 = self.layer2(c1)  # (batch_size, 512, h/8, w/8)
        c3 = self.layer3(c2)  # (batch_size, 1024, h/16, w/16)
        output = self.layer4(c3)  # (shape: (batch_size, 4*512, h/16, w/16))

        # ASPP 모듈 적용
        aspp_out_seg, aspp_out_det = self.aspp(output)  # (batch_size, 2, h/16, w/16)

        low_level_feature = self.low_level_bn(self.low_level_conv(c2))  # (batch_size, 48, h/8, w/8)

        # 업샘플링 및 디코더
        aspp_out_seg = F.interpolate(aspp_out_seg, size=(low_level_feature.size(2), low_level_feature.size(3)),
                                     mode='bilinear',
                                     align_corners=True)  # segmentation feature map decoding

        aspp_out_det = F.interpolate(aspp_out_det, size=(low_level_feature.size(2), low_level_feature.size(3)),
                                     mode='bilinear',
                                     align_corners=True)  # detection feature map decoding

        concat_features = torch.cat([aspp_out_seg, aspp_out_det, low_level_feature], dim=1)  # (batch_size, 2 + 48, h/8, w/8)

        # Decoder Stage
        x = F.relu(self.decoder_bn1(self.decoder_conv1(concat_features)))
        x = F.relu(self.decoder_bn2(self.decoder_conv2(x)))

        # 최종 업샘플링 및 출력
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        output_seg = self.final_conv_seg(x)
        output_det = self.final_conv_det(x)

        return output_seg, output_det

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

