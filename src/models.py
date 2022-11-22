import torch
import torch.nn as nn
import torchvision

from typing import Tuple


def get_encoder(backbone: str) -> Tuple[nn.Module, int]:
    """
    Returns the encoder network for the given backbone as well as the input dimension for the
    pretext-specific network.
    """
    if backbone == "alexnet":
        return AlexNetEncoder(), 8192
    elif "resnet" in backbone:
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18()
            input_dim = 4096
        elif backbone == "resnet34":
            resnet = torchvision.models.resnet34()
            input_dim = 4096
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

        # override the final layers, we are not interested in the
        # fully-connected classifier of the resnet
        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity()

        return resnet, input_dim
    else:
        raise ValueError(f"Invalid backbone: {backbone}")


class AlexNetEncoder(nn.Module):
    """
    Follows the architecture outlined in the patch localization paper.
    Code adopted from: 
        - https://github.com/abhisheksambyal/Self-supervised-learning-by-context-prediction
        - https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html
    """
    def __init__(self):
        super(AlexNetEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(96),
            nn.Conv2d(96, 384, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(384),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fc6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.fc6(x)
        return x


class OriginalPretextNetwork(nn.Module):
    def __init__(self, backbone: str="alexnet"):
        super(OriginalPretextNetwork, self).__init__()
        self.encoder, input_dim = get_encoder(backbone)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 8)
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, 
        center: torch.Tensor, 
        neighbor: torch.Tensor,
        ) -> torch.Tensor:

        # embeddings
        center = self.get_embedding(center)
        neighbor = self.get_embedding(neighbor)

        # pretext task
        output = torch.cat((center, neighbor), 1)
        output = self.fc(output)

        return output


class OurPretextNetwork(nn.Module):
    def __init__(self, backbone: str="alexnet"):
        super(OurPretextNetwork, self).__init__()
        self.encoder, input_dim = get_encoder(backbone)            
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 8)
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, 
        center: torch.Tensor, 
        neighbor1: torch.Tensor, 
        neighbor2: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # embeddings
        center = self.get_embedding(center)
        neighbor1 = self.get_embedding(neighbor1)
        neighbor2 = self.get_embedding(neighbor2)

        # pretext task
        output1 = torch.cat((center, neighbor1), 1)
        output2 = torch.cat((center, neighbor2), 1)
        output1 = self.fc(output1)
        output2 = self.fc(output2)
        
        return output1, output2