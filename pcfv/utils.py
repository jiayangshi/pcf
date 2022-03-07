import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("figure", dpi=250)
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
    TEXTWIDTH_DOUBLE = 455.24408 / 72
    FIGWIDTH_DOUBLE = TEXTWIDTH_DOUBLE
    num_imgs = len(args)
    subposition = kwargs['subposition'] if 'subposition' in kwargs.keys() else (1, num_imgs)

    assert num_imgs <= subposition[0] * subposition[1]

    show_image = kwargs['show_image'] if 'show_image' in kwargs.keys() else True
    assert show_image in (True, False)

    style = kwargs['style'] if 'style' in kwargs.keys() else None
    fig = plt.figure(frameon=True)

    ax1 = plt.subplot(subposition[0], subposition[1], 1)

    vmin = kwargs['vmin'] if 'vmin' in kwargs.keys() else None
    vmax = kwargs['vmax'] if 'vmax' in kwargs.keys() else None
    ax1.imshow(args[0], cmap=style, vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    ax1.set_xlabel(kwargs['x1'] if 'x1' in kwargs.keys() else None)
    ax1.set_title(kwargs['t1'] if 't1' in kwargs.keys() else None)

    ax2 = plt.subplot(subposition[0], subposition[1], 2)
    ax2.imshow(args[1], cmap=style, vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    ax2.set_xlabel(kwargs['x2'] if 'x2' in kwargs.keys() else None)
    ax2.set_title(kwargs['t2'] if 't2' in kwargs.keys() else None)

    if num_imgs >= 3:
        ax3 = plt.subplot(subposition[0], subposition[1], 3)
        ax3.imshow(args[2], cmap=style, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        ax3.set_xlabel(kwargs['x3'] if 'x3' in kwargs.keys() else None)
        ax3.set_title(kwargs['t3'] if 't3' in kwargs.keys() else None)

    if num_imgs >= 4:
        ax4 = plt.subplot(subposition[0], subposition[1], 4)
        ax4.imshow(args[3], cmap=style, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        ax4.set_xlabel(kwargs['x4'] if 'x4' in kwargs.keys() else None)
        ax4.set_title(kwargs['t4'] if 't4' in kwargs.keys() else None)

    if num_imgs >= 5:
        ax5 = plt.subplot(subposition[0], subposition[1], 5)
        ax5.imshow(args[4], cmap=style, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        ax5.set_xlabel(kwargs['x5'] if 'x5' in kwargs.keys() else None)
        ax5.set_title(kwargs['t5'] if 't5' in kwargs.keys() else None)
        
    if num_imgs >= 6:
        ax6 = plt.subplot(subposition[0], subposition[1], 6)
        ax6.imshow(args[5], cmap=style, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        ax6.set_xlabel(kwargs['x6'] if 'x6' in kwargs.keys() else None)
        ax6.set_title(kwargs['t6'] if 't6' in kwargs.keys() else None)

    fig.set_figwidth(FIGWIDTH_DOUBLE)
    fig.set_figheight(FIGWIDTH_DOUBLE)
    if show_image:
        plt.show()
    return fig