import torch
import torch.nn as nn

from layers.ConvBlock import ConvBlock


class UNet(nn.Module):
    '''
    Implementation of UNet (Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation)
    '''
    def __init__(self, in_channels, out_channels):
        '''
        :param in_channels:
        :param out_channels:
        '''
        super(UNet, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.max_pooling3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.max_pooling4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride=2)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=512)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2)
        self.conv_block7 = ConvBlock(in_channels=512, out_channels=256)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        self.conv_block8 = ConvBlock(in_channels=256, out_channels=128)
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)
        self.conv_block9 = ConvBlock(in_channels=128, out_channels=64)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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

        y = self.sigmoid(self.conv1(tmp13))

        return y
