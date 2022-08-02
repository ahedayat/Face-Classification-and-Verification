import torch
import torch.nn as nn


class EuclideanDist(nn.Module):
    """
        Euclidean Distance Module:
            Euclidean-Distance(x_1, x_2) = || x_1 - x_2 ||_2
    """

    def __init__(self):
        super().__init__()
        self.p = 2

        self.dist_fn = nn.PairwiseDistance(p=self.p)

    def forward(self, x1, x2):
        """
            This function calculate Euclidean distance of two tensors.

            Parameters
            ---------------------------------------------------
                - x1 (torch.tensor)
                - x2 (torch.tensor)
        """
        x1_normalized = nn.functional.normalize(x1, p=self.p, dim=1, eps=1e-12)
        x2_normalized = nn.functional.normalize(x1, p=self.p, dim=1, eps=1e-12)
        dist = self.dist_fn(x1_normalized, x2_normalized)

        dist = torch.abs(dist)

        return dist
