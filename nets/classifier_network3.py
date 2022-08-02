import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self, num_cats) -> None:
        super().__init__()

        self.num_cats = num_cats

        self.model = nn.Sequential()

        # 1st Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(10, 10))
        self.act1 = nn.ReLU()
        self.pad1 = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2))

        # 2nd Block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(7, 7))
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2))

        # 3rd Block
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(4, 4))
        self.act3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((2, 2))

        # Final Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4))
        self.act4 = nn.ReLU()

        # Linaer Layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(9216, 4096)
        self.act_linear = nn.Sigmoid()

        # Classification Layer
        self.classification = nn.Linear(4096, self.num_cats)
        self.act_classification = nn.Sigmoid()

    def padding(self, _in):
        """
            Padding _in, if height or width of a tensor is odd.
        """
        out = _in

        if out.shape[2] % 2 == 1:
            out = F.pad(out, (0, 0, 0, 1))
        if out.shape[3] % 2 == 1:
            out = F.pad(out, (0, 1, 0, 0))

        return out

    def forward(self, _in):
        """
            Forwarding input to output
        """

        # 1st Block
        out = self.act1(self.conv1(_in))
        out = self.padding(out)
        out = self.maxpool1(out)

        # 2nd Block
        out = self.act2(self.conv2(out))
        out = self.padding(out)
        out = self.maxpool2(out)

        # 3rd Block
        out = self.act3(self.conv3(out))
        out = self.padding(out)
        out = self.maxpool3(out)

        # Final Block
        out = self.act4(self.conv4(out))

        # Linear Layer
        out = self.flatten(out)

        out = self.linear(out)
        out = self.act_linear(out)

        # Classification Layer
        out = self.classification(out)
        out = self.act_classification(out)

        return out
