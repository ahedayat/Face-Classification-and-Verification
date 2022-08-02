import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self, backbone, num_cats, embeddiing_size=512) -> None:
        super().__init__()

        self.num_cats = num_cats
        self.embeddiing_size = embeddiing_size

        self.backbone = backbone

        # After Last Convoloution
        self.backbone.layer4 = nn.Sequential(
            self.backbone.layer4,
            nn.Dropout(
                p=0.5, inplace=False),
            nn.Flatten(),
            nn.Linear(32768, self.embeddiing_size),
            nn.ReLU()
        )

        # Average Pool
        self.backbone.avgpool = nn.AdaptiveAvgPool1d(self.embeddiing_size)

        # Classification Layer
        self.backbone.fc = nn.Linear(self.embeddiing_size, self.num_cats)

        self.softmax = nn.Softmax(dim=1)

        # self.act_classification = nn.Sigmoid()

    def forward(self, _in):
        """
            Forwarding input to output
        """
        # print("******** Before fc: {}".format(out.shape))
        out = self.backbone(_in)
        out = self.softmax(out)

        return out

    def embedding(self):
        """
            This function replace two final layer with nn.Identify to extract 
            a embedding for input
        """
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
