import numpy as np

def add_possion_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img /= photon_count
    img = -np.log(img, **opt)
    return img