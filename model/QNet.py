import torch
import torchvision.models as models
class Qnet(torch.nn.Module):
    ''' determinstic '''
    def __init__(self, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # self.conv1 = torch.nn.Conv2d(state_dim, hidden_dim,3,2,1)
        self.resNet = models.resnet50(pretrained=True)
        num_ftrs = self.resNet.fc.in_features
        self.resNet.fc = torch.nn.Linear(num_ftrs, hidden_dim)
        for param in self.resNet.parameters():
            param.requires_grad = False
        for param in self.resNet.layer4.parameters():
            param.requires_grad = True
        self.resNet.fc.requires_grad = True
        self.fc1 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.resNet(x)
        x = self.fc1(x)
        return x