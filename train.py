import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device, dtype=torch.float)
        pred = model(X)
        label = y.to(device, dtype=torch.float)
        loss = loss(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # display current batch and loss
        loss, current = loss.item(), batch * len(X)
        print(f"training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # calculates the average training loss
        training_loss += loss/batches

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
    valid_loss = 0
    size = len(dataloader.dataset)
    batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device, dtype=torch.float)
        pred = model(X)
        label = y.to(device, dtype=torch.float)
        loss = loss(pred, label)

        # display current batch and loss
        loss, current = loss.item(), batch * len(X)
        print(f"validation loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # calculates the average training loss
        valid_loss += loss/batches
    return valid_loss

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
            loss = loss(pred, label)
            metric = metric(pred, label)
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
            test_loss += loss / batches
            test_metric += metric / batches

    print(f"Avg loss on whole image: {test_loss:>8f} \n")
    print(f"Avg metric on whole image: {test_metric:>8f} \n")