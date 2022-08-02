import torch
import torch.nn as nn


class L1Dist(nn.Module):
    """
        L1 Distance Module:
            L1-Distance(x_1, x_2) = | x_1 - x_2 |
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x1, x2):
        """
            This function calculate L1 Distance of two tensors.

            Parameters
            ---------------------------------------------------
                - x1 (torch.tensor)
                - x2 (torch.tensor)
        """
        dist = torch.abs(x1, x2)
        return dist
