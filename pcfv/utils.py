import matplotlib.pyplot as plt

'''
The function used to count the trainable parameters of a network
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_images(*args, **kwargs):
    '''
    :param args: The images to plot, number can be either 2 or 3
    :param kwargs: 't1' specifies the title of first image
                   'x1' specifies the title of first image
                   'direction' specifies the subplot direction of images
                   'style' specifies the image style
    :return:
    '''
    num_imgs = len(args)
    assert num_imgs in [2,3]
    direction = 'horizontal'
    if 'direction' in kwargs.keys():
        direction = kwargs['direction']
        assert direction in ['horizontal', 'vertical']
    style = kwargs['style'] if 'style' in kwargs.keys() else None
    fig = plt.figure(frameon=True)
    if direction=='horizontal':
        ax1 = plt.subplot(1, num_imgs, 1)
    else:
        ax1 = plt.subplot(num_imgs, 1, 1)
    ax1.imshow(args[0], cmap=style)
    plt.xticks([])
    plt.yticks([])
    ax1.set_xlabel(kwargs['x1'] if 'x1' in kwargs.keys() else None)
    ax1.set_title(kwargs['t1'] if 't1' in kwargs.keys() else None)

    if direction=='horizontal':
        ax2 = plt.subplot(1, num_imgs, 2)
    else:
        ax2 = plt.subplot(num_imgs, 1, 2)
    ax2.imshow(args[1], cmap=style)
    plt.xticks([])
    plt.yticks([])
    ax2.set_xlabel(kwargs['x2'] if 'x2' in kwargs.keys() else None)
    ax2.set_title(kwargs['t2'] if 't2' in kwargs.keys() else None)

    if num_imgs == 3:
        if direction=='horizontal':
            ax3 = plt.subplot(1, num_imgs, 3)
        else:
            ax3 = plt.subplot(num_imgs, 1, 3)
        ax3.imshow(args[2], cmap=style)
        plt.xticks([])
        plt.yticks([])
        ax3.set_xlabel(kwargs['x3'] if 'x3' in kwargs.keys() else None)
        ax3.set_title(kwargs['t3'] if 't3' in kwargs.keys() else None)
    plt.show()