import torch.nn as nn
from torch import sigmoid, tanh
from torch import cat
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim, cond_dim, out_channels):
        super(Generator, self).__init__()
        self.lin4cond = nn.Linear(cond_dim, 100)
        self.lin1 = nn.Linear(input_dim + 100, 256 * 8 * 8)
        self.upsample = nn.Upsample(scale_factor=2)
        self.batchnorm1 = nn.BatchNorm2d(256, momentum=0.8)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128, momentum=0.8)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64, momentum=0.8)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x_in, cond):
        cond = self.lin4cond(cond.float())
        x = cat((x_in, cond), dim=2).squeeze()
        x = self.lin1(x)
        x = x.view(-1, 256, 8, 8)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_dim, cond_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(image_dim[2], 64, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.batchnorm1 = nn.BatchNorm2d(128, momentum=0.8)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256, momentum=0.8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(512, momentum=0.8)
        self.conv5 = nn.Conv2d(612, 512, kernel_size=3, stride=1, padding=1)

        self.lin4cond = nn.Linear(cond_dim, 100)
        self.finallin = nn.Linear(9 * 9 * 512, 1)

    def forward(self, image, cond):
        x = self.conv1(image)
        x = nn.LeakyReLU()(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.zeropad(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        cond = self.lin4cond(cond.float())
        cond = cond.view(-1, 100, 1, 1)
        cond = cond.repeat(1, 1, 9, 9)

        x = cat((x, cond), dim=1)
        x = self.conv5(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.finallin(x)
        x = sigmoid(x)
        return x

