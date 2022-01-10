import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv3d=False):
        super(ConvBlock, self).__init__()
        if conv3d:
            Conv =  nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d

        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.batch_norm1 = BatchNorm(num_features=out_channels)
        self.batch_norm2 = BatchNorm(num_features=out_channels)

    def forward(self, x):
        tmp = self.relu1(self.batch_norm1(self.conv1(x)))
        y = self.relu2(self.batch_norm2(self.conv2(tmp)))
        return y
