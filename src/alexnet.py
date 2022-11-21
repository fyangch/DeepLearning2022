import torch
import torch.nn as nn

"""
Follows the architecture outlined in the patch localization paper
Code adopted from: 
    - https://github.com/abhisheksambyal/Self-supervised-learning-by-context-prediction
    - https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html
"""
class EncoderNetwork(nn.Module):
    def __init__(self):
        super(EncoderNetwork, self).__init__()
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
            nn.Flatte(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc6(x)
        return x


class OriginalPretextNetwork(nn.Module):
    def __init__(self, aux_logits=False):
        super(OriginalPretextNetwork, self).__init__()
        self.encoder = EncoderNetwork()
        self.fc = nn.Sequential(
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 8)
        )

    def get_embedding(self, x):
        return self.encoder(x)

    def forward(self, center, neighbor):
        # embeddings
        center = self.get_embedding(center)
        neighbor = self.get_embedding(neighbor)

        # pretext task
        output = torch.cat((center, neighbor), 1)
        output = self.fc(output)

        return output


class OurPretextNetwork(nn.Module):
    def __init__(self, aux_logits=False):
        super(OriginalPretextNetwork, self).__init__()
        self.encoder = EncoderNetwork()
        self.fc = nn.Sequential(
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 8)
        )

    def get_embedding(self, x):
        return self.encoder(x)

    def forward(self, center, neighbor1, neighbor2):
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