import torch
import torchvision.models as models
from torch import nn


class Qnet(torch.nn.Module):
    ''' determinstic '''
    def __init__(self, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # self.conv1 = torch.nn.Conv2d(state_dim, hidden_dim,3,2,1)
        self.resNet = models.resnet50(pretrained=False)
        num_ftrs = self.resNet.fc.in_features
        self.resNet.fc = torch.nn.Linear(num_ftrs, hidden_dim)
        for param in self.resNet.parameters():
            param.requires_grad = True
        # for param in self.resNet.layer4.parameters():
        #     param.requires_grad = True
        # self.resNet.fc.requires_grad = True
        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, action_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu1(self.resNet(x))
        print(f'x.shape: {x.shape}')
        x = self.fc1(x)
        print(f'x.shape: {x.shape}')
        return x