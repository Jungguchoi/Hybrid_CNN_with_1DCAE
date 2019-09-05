import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim

import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms


## 1D Convolutional autoencoder model with hybrid cnn
class Conv_AE(nn.Module):
    def __init__(self):
        super(Conv_AE, self).__init__()

        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 8, 30, 2),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )

        self.Encoder = nn.Sequential(
            nn.Conv1d(8, 16, 20, 2),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, 32, 10, 2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Conv1d(32, 64, 10, 1),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Conv1d(64, 128, 10, 1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 10, 1),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.ConvTranspose1d(64, 32, 10, 1),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.ConvTranspose1d(32, 16, 10, 2),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.ConvTranspose1d(16, 8, 20, 2),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.ConvTranspose1d(8, 1, 30, 2),
            nn.Tanh()
        )

        self.linear1 = nn.Linear(720, 720)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def hybrid_layer(self, x_activity, x_covariate):
        x_activity = x_activity.view(-1, 1, 720)

        covariate_weight = torch.randn((x_activity.size()[0], 5), requires_grad=True).double().cuda()

        demo_tensor = x_covariate.cuda()
        demo_tensor.requires_grad=False

        weight_sum_result = torch.sum(demo_tensor * covariate_weight)
        self.initial_layer_weight.data = self.initial_layer.weight.add(weight_sum_result)

        after_x = self.initial_layer(x_activity)

        return after_x

    def initial_layer(self, x):

        ## Split activity data and covariate data
        x_activity = x[:720]
        x_covariate = x[-7:-3]
        x_activity = x_activity.view(-1, 1, 720)
        x_covariate = x_covariate.view(-1, 1, 4)

        after_x = self.hybrid_layer(x_activity, x_covariate)

        return after_x


    def encoder(self, x):
        x_ = self.Encoder(x).view(-1, 720)

        return x_

    def decoder(self, x):
        x = self.Decoder(x).view(-1, 720)

        return x

    def forward(self, x):
        x = x.view(-1, 1, len(x))
        after_x = self.initial_layer(x)
        encode = self.encoder(after_x)
        decode = self.decoder(encode)
        return decode.view(-1, 720)