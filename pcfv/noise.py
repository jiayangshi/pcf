import numpy as np

def transmittance(sinogram):
    return np.mean(np.exp(-sinogram)[sinogram > 0])

def absorption(sinogram):
    return 1 - transmittance(sinogram)

def add_possion_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img = np.divide(img, photon_count, **opt)
    img = -np.log(img, **opt)
    return img

def cal_attenuation_factor(sinogram, target, margin):
    print("Calculating the attenuation factor")
    factor = 1
    while (abs(absorption(sinogram*factor)*100 - target) > margin):
        if absorption(sinogram*factor)*100 > target:
            factor /= 10
            if (absorption(sinogram*factor)*100 < target):
                factor *= 2
        else:
            factor *= 10
            if (absorption(sinogram*factor)*100 > target):
                factor /= 3
    return factor


