import torch
import numpy  as np

def psnr(output, label):
    '''
    :param output:
    :param label:
    :return: PSNR value calculated in pytorch way
    '''
    mse = torch.mean((output - label) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))

def psnr_np(output, label):
    '''
    :param output:
    :param label:
    :return: PSNR value calculated in pytorch way
    '''
    mse = np.mean((output - label) ** 2)
    return 20 * np.log10(1 / np.sqrt(mse))