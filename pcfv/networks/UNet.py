import torch
import torch.nn as nn

from pcfv.layers.ConvBlock import ConvBlock
from pcfv.layers.ScalingBlock import ScalingBlock

class UNet(nn.Module):
    '''
    Implementation of UNet (Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation)
    '''
    def __init__(self, in_channels, out_channels, inter_channel=64):
        '''
        :param in_channels:
        :param out_channels:
        '''
        super(UNet, self).__init__()

        self.scale_in = ScalingBlock(in_channels)
        self.scale_out = ScalingBlock(out_channels)

        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=inter_channel)
        self.conv_block2 = ConvBlock(in_channels=inter_channel, out_channels=inter_channel*2)
        self.conv_block3 = ConvBlock(in_channels=inter_channel*2, out_channels=inter_channel*4)
        self.conv_block4 = ConvBlock(in_channels=inter_channel*4, out_channels=inter_channel*8)
        self.conv_block5 = ConvBlock(in_channels=inter_channel*8, out_channels=inter_channel*16)
        self.conv_block6 = ConvBlock(in_channels=inter_channel*16, out_channels=inter_channel*8)
        self.conv_block7 = ConvBlock(in_channels=inter_channel*8, out_channels=inter_channel*4)
        self.conv_block8 = ConvBlock(in_channels=inter_channel*4, out_channels=inter_channel*2)
        self.conv_block9 = ConvBlock(in_channels=inter_channel*2, out_channels=inter_channel)

        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=inter_channel*16, out_channels=inter_channel*8, kernel_size=2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=inter_channel*8, out_channels=inter_channel*4, kernel_size=2, stride=2)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=inter_channel*4, out_channels=inter_channel*2, kernel_size=2, stride=2)
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=inter_channel*2, out_channels=inter_channel, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(in_channels=inter_channel, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.scale_in(x)

        tmp1 = self.conv_block1(x)
        tmp2 = self.conv_block2(self.max_pooling1(tmp1))
        tmp3 = self.conv_block3(self.max_pooling1(tmp2))
        tmp4 = self.conv_block4(self.max_pooling1(tmp3))

        tmp5 = self.conv_block5(self.max_pooling1(tmp4))

        tmp6 = self.conv_transpose1(tmp5)
        tmp7 = self.conv_block6(torch.cat((tmp6, tmp4), dim=1))
        tmp8 = self.conv_transpose2(tmp7)
        tmp9 = self.conv_block7(torch.cat((tmp8, tmp3), dim=1))
        tmp10 = self.conv_transpose3(tmp9)
        tmp11 = self.conv_block8(torch.cat((tmp10, tmp2), dim=1))
        tmp12 = self.conv_transpose4(tmp11)
        tmp13 = self.conv_block9(torch.cat((tmp12, tmp1), dim=1))

        y = self.final_conv(tmp13)
        y = self.scale_out(y)

        return y

    def normalized_input(self, x):
        x = self.scale_in(x)
        return x