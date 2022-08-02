import torch
import torch.nn as nn
import torch.nn.functional as F
from .last_layers import ArcFaceLayer


class ClassificationNetwork(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self, num_cats, resnet, s=30, m=0.5, device_last_layer=torch.device("cpu"), train=True) -> None:
        super().__init__()

        self.num_cats = num_cats
        self._train = train

        self.resnet = resnet

        # Last layer must be a ArcFace layer
        self.s = s
        self.m = m

        # import pdb
        # pdb.set_trace()

        self.last_layer = ArcFaceLayer(
            in_features=self.resnet.fc5.out_features,
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
        if self._train:
            out = self.last_layer(out, label)

        return out

    def embedding(self):
        """
            This function replace two final layer with nn.Identify to extract 
            a embedding for input
        """
        self.backbone.avgpool = nn.Identity()

        self.backbone.fc = nn.Identity()

    def _train(self):
        self._train = True

    def _eval(self):
        self._train = False
