import torch
import math
import torch.nn as nn
import numpy as np

from layers.MSDConvBlock import MSDConvBlock
from layers.ScalingBlock import ScalingBlock, scaling_module_set_scale, scaling_module_set_bias
torch.autograd.set_detect_anomaly(True)

'''
Implementation of MSDNet with pytorch (Pelt A mixed-scale dense convolutional neural network for image analysis.)
'''
class MSDNet(nn.Module):
    def __init__(self, in_channels, out_channels, width, depth, dilations=None):
        '''
        :param in_channels: input channels
        :param out_channels: output channels
        :param width: the width of MSD network, for this implementation it is fixed at 1
        :param depth: the depth of MSD network
        :param dilations: the used dilations of convolutions
        '''
        super(MSDNet, self).__init__()
        if dilations is None:
            dilations = [i + 1 for i in range(10)]
        if dilations is None:
            dilations = [i + 1 for i in range(10)]
        assert width == 1
        self.width = 1
        self.depth = depth
        self.shared_bias  = nn.ParameterList([nn.Parameter(torch.zeros(1,)) for _ in range(depth)])
        if dilations is None:
            self.dilations = [i + 1 for i in range(10)]
        else:
            self.dilations = dilations
        self.hiddenstates = [None for i in range(depth)]
        self.msdlayers = []
        self.counter = 0
        for i in range(depth):
            for z in range(1+i*width):
                if z==0:
                    self.msdlayers.append(
                        MSDConvBlock(in_channels, 1, self.dilations[self.counter % len(self.dilations)],
                                     False, math.sqrt(2 / (9 * (in_channels + width * (depth - 1)) + out_channels))))
                else:
                    self.msdlayers.append(
                        MSDConvBlock(1, 1, self.dilations[self.counter%len(self.dilations)],
                        False, math.sqrt(2/(9*(in_channels+width*(depth-1))+out_channels))))
            self.counter += 1

        self.relu = nn.ReLU(inplace=True)
        self.convlayer = nn.Conv2d(in_channels=width * depth + in_channels, out_channels=out_channels, kernel_size=(1, 1))
        torch.nn.init.zeros_(self.convlayer.weight)
        torch.nn.init.zeros_(self.convlayer.bias)

        self.layers = nn.ModuleList(self.msdlayers)
        self.scale_in = ScalingBlock(in_channels, conv3d=False)
        self.scale_out = ScalingBlock(out_channels, conv3d=False)

    def forward(self, x):
        x = self.scale_in(x)
        counter = 0
        for i in range(self.depth):
            for z in range(1 + i * self.width):
                if z == 0:
                    self.hiddenstates[i] = self.msdlayers[counter](x)
                else:
                    self.hiddenstates[i] += self.msdlayers[counter](self.hiddenstates[z - 1])
                if z == i * self.width:
                    self.hiddenstates[i] = self.relu(torch.add(self.hiddenstates[i], self.shared_bias[i]))
                counter += 1
        tmp = x

        for i in range(self.depth):
            tmp = torch.cat((tmp, self.hiddenstates[i]), dim=1)

        y = self.convlayer(tmp)
        y = self.scale_out(y)
        return y

    def set_normalization(self, dataloader):
        """Normalize input and target data.

        This function goes through all the training data to compute
        the mean and std of the training data.

        It modifies the network so that all future invocations of the
        network first normalize input data and target data to have
        mean zero and a standard deviation of one.

        These modified parameters are not updated after this step and
        are stored in the network, so that they are not lost when the
        network is saved to and loaded from disk.

        Normalizing in this way makes training more stable.

        :param dataloader: The dataloader associated to the training data.
        :returns:
        :rtype:

        """
        mean_in = square_in = mean_out = square_out = 0

        for (data_in, data_out) in dataloader:
            mean_in += data_in.mean()
            mean_out += data_out.mean()
            square_in += data_in.pow(2).mean()
            square_out += data_out.pow(2).mean()

        mean_in /= len(dataloader)
        mean_out /= len(dataloader)
        square_in /= len(dataloader)
        square_out /= len(dataloader)

        std_in = np.sqrt(square_in - mean_in ** 2)
        std_out = np.sqrt(square_out - mean_out ** 2)

        # The input data should be roughly normally distributed after
        # passing through scale_in. Note that the input is first
        # scaled and then recentered.
        scaling_module_set_scale(self.scale_in, 1 / std_in)
        scaling_module_set_bias(self.scale_in, -mean_in / std_in)
        # The scale_out layer should rather 'denormalize' the network
        # output.
        scaling_module_set_scale(self.scale_out, std_out)
        scaling_module_set_bias(self.scale_out, mean_out)





