from typing import Optional, Any, Tuple
import numpy
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torch.autograd import Function
from mixstyle import MixStyle

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Classifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=2):
        super(Classifier, self).__init__()

        self.classifier_layer = nn.Linear(in_channels, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if(norm_flag): # norm classification weight
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            output = self.classifier_layer(input)
        else:
            output = self.classifier_layer(input)
        return output

class BaseModel(nn.Module):
    def __init__(self,
                model_name='resnet18',
                pretrained=False,
                num_classes=2):
        super(BaseModel, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='max')
        #print(self.feature_extractor.default_cfg)
        in_channels =  self.feature_extractor.feature_info[-1]['num_chs']
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

    def forward(self, input):
        feature = self.feature_extractor(input)
        cls = self.classifier(feature)
        return cls

class BaseMixModel(nn.Module):
    def __init__(self,
                model_name='resnet18',
                pretrained=False,
                num_classes=2,
                ms_class=MixStyle, #None,
                ms_layers=["layer1", "layer2"],
                ms_p=0.5,
                ms_a=0.1,
                mix="crossdomain",):
        super(BaseMixModel, self).__init__()

        feature_extractor = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        features = nn.ModuleList(feature_extractor.children())
        self.conv1 = torch.nn.Sequential(*features[:-4])
        self.layer1 = features[-4]
        self.layer2 = features[-3]
        self.layer3 = features[-2]
        self.layer4 = features[-1]

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        in_channels =  feature_extractor.feature_info[-1]['num_chs']
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a, mix=mix)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3", "layer4"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
            print(f"Using {mix}")
        else:
            print("No MixStyle used")
        self.ms_layers = ms_layers

    def forward(self, input):
        x = self.conv1(input)

        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer4(x)
        if "layer4" in self.ms_layers:
            x = self.mixstyle(x)

        x = self.avgpool(x)
        emb = x.view(x.size(0), -1)

        x = self.classifier(emb)

        return x

def _test():
    import torch
    labels = torch.tensor([0, 1, 1, 0])
    image_x = torch.randn(16, 3, 224, 224)

    model = BaseMixModel(model_name='resnet18')
    y = model(image_x)
    print(y.shape)

if __name__ == "__main__":
    _test()
