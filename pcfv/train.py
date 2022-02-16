import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .layers.ScalingBlock import scaling_module_set_scale, scaling_module_set_bias

def train_loop(dataloader, model, optimizer, loss, device='cuda'):
    '''
    The training function
    :param dataloader: The dataloader to provide the data
    :param model: The to be trained model
    :param optimizer: The optimizer
    :param loss: The used loss function
    :param device: The device on which the training is done
    :return: The average training loss
    '''
    training_loss = 0
    size = len(dataloader.dataset)
    batches = len(dataloader)
    bar = tqdm(dataloader)
    for batch, (X, y) in enumerate(bar):
        # Compute prediction and loss
        X = X.to(device, dtype=torch.float)
        pred = model(X)
        label = y.to(device, dtype=torch.float)
        cur_loss = loss(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        # display current batch and loss
        cur_loss, current = cur_loss.item(), batch * len(X)
        bar.set_description(f"training loss: {cur_loss:>7f}  [{current:>5d}/{size:>5d}]")

        # calculates the average training loss
        training_loss += cur_loss/batches

    return training_loss

def valid_loop(dataloader, model, loss, device='cuda'):
    '''
    The validation function
    :param dataloader: The dataloader to provide the data
    :param model: The to be trained model
    :param loss: The used loss function
    :param device: The device on which the validation is done
    :return: The average validation loss
    '''
    validation_loss = 0
    size = len(dataloader.dataset)
    batches = len(dataloader)
    bar = tqdm(dataloader)
    with torch.no_grad():
        for batch, (X, y) in enumerate(bar):
            # Compute prediction and loss
            X = X.to(device, dtype=torch.float)
            pred = model(X)
            label = y.to(device, dtype=torch.float)
            cur_loss = loss(pred, label)

            # display current batch and loss
            cur_loss, current = cur_loss.item(), batch * len(X)
            bar.set_description(f"validation loss: {cur_loss:>7f}  [{current:>5d}/{size:>5d}]")

            # calculates the average training loss
            validation_loss += cur_loss/batches
    return validation_loss

def test_loop(dataloader, model, loss, metric, output_directory=None,  device='cuda'):
    '''

    :param dataloader: The dataloader to provide the data
    :param model: The to be trained model
    :param loss: The used loss function
    :param metric: The used metric function
    :param output_directory: The directory to save the test results
    :param device: The device on which the validation is done
    '''
    batches = len(dataloader)
    test_loss, test_metric = 0, 0

    i = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, dtype=torch.float)
            pred = model(X)
            label = y.to(device, dtype=torch.float)
            cur_loss = loss(pred, label)
            cur_metric = metric(pred, label)
            if not output_directory:
                for j in range(pred.shape[0]):
                    fig = plt.figure(frameon=True)
                    ax1 = plt.subplot(1, 3, 1)
                    ax1.imshow(np.squeeze(label[j].cpu().numpy()), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    ax1.set_title("Original")
                    ax2 = plt.subplot(1, 3, 2)
                    ax2.imshow(np.squeeze(X[j].cpu().numpy()), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    ax2.set_title("Noised")
                    ax2.set_xlabel("PSNR:{:,.2f} dB".format(metric(label[j], X[j]).cpu().numpy()))
                    ax3 = plt.subplot(1, 3, 3)
                    ax3.imshow(np.squeeze(pred[j].cpu().numpy()), vmin=0, vmax=1)
                    plt.xticks([])
                    plt.yticks([])
                    ax3.set_title("Denoised")
                    ax3.set_xlabel("PSNR:{:,.2f} dB".format(metric(label[j], X[j]).cpu().numpy()))
                    fig.savefig(os.path.join(output_directory, str(i) + ".png"))
                    print("The {}th test image is processed".format(i + 1))
                    i += 1
            test_loss += cur_loss / batches
            test_metric += cur_metric / batches

    print(f"Avg loss on whole image: {test_loss:>8f} \n")
    print(f"Avg metric on whole image: {test_metric:>8f} \n")


def set_normalization(model, dataloader):
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
    print("Calculating the normalization factors")
    mean_in = square_in = mean_out = square_out = 0

    for (data_in, data_out) in dataloader:
        mean_in += data_in.mean(axis=(2,3))
        mean_out += data_out.mean(axis=(2,3))
        square_in += data_in.pow(2).mean(axis=(2,3))
        square_out += data_out.pow(2).mean(axis=(2,3))

    mean_in /= len(dataloader)
    mean_out /= len(dataloader)
    square_in /= len(dataloader)
    square_out /= len(dataloader)

    std_in = np.sqrt(square_in - mean_in ** 2)
    std_out = np.sqrt(square_out - mean_out ** 2)

    # The input data should be roughly normally distributed after
    # passing through scale_in. Note that the input is first
    # scaled and then recentered.
    scaling_module_set_scale(model.scale_in, 1 / std_in)
    scaling_module_set_bias(model.scale_in, -mean_in / std_in)
    # The scale_out layer should rather 'denormalize' the network
    # output.
    scaling_module_set_scale(model.scale_out, std_out)
    scaling_module_set_bias(model.scale_out, mean_out)

def early_stopping(valid_losses, patience=4):
    if len(valid_losses) > patience:
        # if current loss larger than max value in patience
        # or last patience number losses non decreasing
        if valid_losses[-1] > max(valid_losses[-patience-1:-1]) or \
                all(x<=y for x, y in zip(valid_losses[-patience-1:-1], valid_losses[-patience:])):
            return True
    return False

