import torch
import torch.nn as nn
import torch.nn.functional as F
from .last_layers import ArcFaceLayer
from .net_utils import l2_norm
from torchvision import models


class ClassificationTrainNetwork(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self, num_cats, resnet, s=30, m=0.5, device_last_layer=torch.device("cpu"), train=True) -> None:
        super().__init__()

        self.num_cats = num_cats
        self.arc_layer = train

        self.resnet = resnet

        # Last layer must be a ArcFace layer
        self.s = s
        self.m = m

        # import pdb
        # pdb.set_trace()
        last_layer_in_features = None
        if isinstance(resnet, models.resnet.ResNet):
            self.resnet.fc = nn.Linear(
                in_features=self.resnet.fc.in_features, out_features=512, bias=True)

            last_layer_in_features = self.resnet.fc.out_features

        else:
            last_layer_in_features = self.resnet.fc5.out_features

        self.last_layer = ArcFaceLayer(
            in_features=last_layer_in_features,
            out_features=self.num_cats,
            s=self.s,
            m=self.m,
            _device=device_last_layer
        )
        # Disabling ResNet Classification Layer
        # self.resnet.fc = nn.Identity()

    def forward(self, _in, label):
        """
            Forwarding input to output
        """
        # print("******** Before fc: {}".format(out.shape))
        out = self.resnet(_in)
        out = l2_norm(out)
        if self.arc_layer:
            out = self.last_layer(out, label)

        return out

    def feature(self, _in):
        """
            Return 512-D feature of `_in` images
        """
        _feature = self.resnet(_in)

        return l2_norm(_feature)

    def embedding(self):
        """
            This function replace two final layer with nn.Identify to extract 
            a embedding for input
        """
        self.backbone.avgpool = nn.Identity()

        self.backbone.fc = nn.Identity()

    def activate_arc_layer(self):
        self.arc_layer = True

    def deactivate_arc_layer(self):
        self.arc_layer = False
