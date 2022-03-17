import torch
import torch.nn as nn

def ScalingBlock(num_channels):
    """Make a Module that normalizes the input data.

    This part of the network can be used to renormalize the input
    data. Its parameters are

    * saved when the network is saved;
    * not updated by the gradient descent solvers.

    :param num_channels: The number of channels.
    :param conv3d: Indicates that the input data is 3D instead of 2D.
    :returns: A scaling module.
    :rtype: torch.nn.ConvNd

    """
    c = nn.Conv2d(num_channels, num_channels, 1)
    c.bias.requires_grad = False
    c.weight.requires_grad = False

    scaling_module_set_scale(c, 1.0)
    scaling_module_set_bias(c, 0.0)

    return c


def scaling_module_set_scale(sm, s):
    device = sm.weight.data.device
    c_out, c_in = sm.weight.shape[:2]
    assert c_out == c_in
    if torch.is_tensor(s):
        s = s.float()
    if isinstance(s, float) or (torch.is_tensor(s) and s.shape[1]==1):
        sm.weight.data.zero_()
        for i in range(c_out):
            sm.weight.data[i, i] = s
    if torch.is_tensor(s) and s.shape[1]>1:
        assert c_out == c_in == s.shape[1]
        sm.weight.data.zero_()
        for i in range(c_out):
            sm.weight.data[i, i] = s[:, i]
    sm.to(device)


def scaling_module_set_bias(sm, bias):
    device = sm.bias.data.device
    if isinstance(bias, float) :
        sm.bias.data.fill_(bias)
    if torch.is_tensor(bias):
        bias = bias.float()
        sm.bias.data = bias[0]
    sm.to(device)
