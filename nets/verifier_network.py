import torch.nn as nn
import torch.nn.functional as F


class VerifierNetwork(nn.Module):
    """
        In this module two faces is verified to be the same or different.
    """

    def __init__(self, embedding_net, embedding_size, dist) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.dist = dist

        # Embedding Network
        self.embedding_net = embedding_net

        # Last Linear Layer
        self.linear = nn.Linear(self.embedding_size, 1)
        self.linear_act = nn.Sigmoid()

    def forward(self, x_1, x_2):
        """
            This method return that two faces are the same or different.
        """
        y_1 = self.embedding_net(x_1)
        y_2 = self.embedding_net(x_2)

        _dist = self.dist(y_1, y_2)

        verified = self.linear_act(self.linear(_dist))

        return verified
