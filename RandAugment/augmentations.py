# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


# transforms

# Cell
from scipy.interpolate import CubicSpline
from scipy.ndimage import convolve1d
import pywt
import torchvision.transforms as transforms


# Cell
@patch
def mul_min(x:(torch.Tensor, TSTensor, NumpyTensor), axes=(), keepdim=False):
    if axes == (): return retain_type(x.min(), x)
    axes = reversed(sorted(axes if is_listy(axes) else [axes]))
    min_x = x
    for ax in axes: min_x, _ = min_x.min(ax, keepdim)
    return retain_type(min_x, x)

@patch
def mul_max(x:(torch.Tensor, TSTensor, NumpyTensor), axes=(), keepdim=False):
    if axes == (): return retain_type(x.max(), x)
    axes = reversed(sorted(axes if is_listy(axes) else [axes]))
    max_x = x
    for ax in axes: max_x, _ = max_x.max(ax, keepdim)
    return retain_type(max_x, x)


# smooth random curve
def random_curve_generator(o, magnitude=.1, order=4, noise=None):
    seq_len = o.shape[-1]
    f = CubicSpline(np.linspace(-seq_len, 2 * seq_len - 1, 3 * (order - 1) + 1, dtype=int),
                    np.random.normal(loc=1.0, scale=magnitude, size=3 * (order - 1) + 1), axis=-1)
    return f(np.arange(seq_len))

def random_cum_curve_generator(o, magnitude=.1, order=4, noise=None):
    x = random_curve_generator(o, magnitude=magnitude, order=order, noise=noise).cumsum()
    x -= x[0]
    x /= x[-1]
    x = np.clip(x, 0, 1)
    return x * (o.shape[-1] - 1)

def random_cum_noise_generator(o, magnitude=.1, noise=None):
    seq_len = o.shape[-1]
    x = np.clip(np.ones(seq_len) + np.random.normal(loc=0, scale=magnitude, size=seq_len), 0, 1000).cumsum()
    x -= x[0]
    x /= x[-1]
    return x * (o.shape[-1] - 1)

def maddest(d, axis=None): #Mean Absolute Deviation
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# transforms
def Identity(img, v):
    "Applies the identity tfm to a batch"
    return img

def MagNoise(img, v):
    "Applies additive noise on the y-axis for each step of a batch"
    return img + torch.normal(0, v, (1, img.shape[-1]), dtype=img.dtype, device=img.device)

def MagMulNoise(img, v):
    "Applies multiplicative noise on the y-axis for each step of a batch"
    return img * torch.normal(0, v, (1, img.shape[-1]), dtype=img.dtype, device=img.device)

def TimeNoise(img, v):
    "Applies noise to each step in the x-axis of a batch based on smooth random curve"
    f= CubicSpline(np.arange(img.shape[-1]), img.cpu(), axis=-1)
    return img.new_tensor(f(random_cum_noise_generator(img, magnitude=v)))

def MagWarp(img, v):
    "Applies warping to the y-axis of a batch based on a smooth random curve"
    y_mult = random_curve_generator(img, magnitude=v)
    return img * img.new_tensor(y_mult)

def TimeWarp(img, v):
    "Applies time warping to the x-axis of a batch based on a smooth random curve"
    f=CubicSpline(np.arange(img.shape[-1]), img.cpu(), axis=-1)
    return img.new_tensor(f(random_cum_curve_generator(img, magnitude=v)))

def MagScale(img, v):
    "Applies scaling to each step in the y-axis of a batch based on a smooth random curve"
    scale = 1 + 2 * (torch.rand(1, device=img.device) - .5) * v
    return img * scale

def MagScaleVar(img, v):
    "Applies scaling to each variable and step in the y-axis of a batch based on smooth random curves"
    scale = 1 + 2 * (torch.rand((img.shape[-2], 1), device=img.device) - .5) * v
    return img * scale

def ZoomIn(img, v):
    "Amplifies a sequence focusing on a random section of the steps"
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = max(lambd, 1 - lambd)
    win_len = int(seq_len * lambd)
    start = 0 if win_len == seq_len else np.random.randint(0, seq_len - win_len)
    f = CubicSpline(np.arange(win_len), img[..., start : start + win_len].cpu(), axis=-1)
    return img.new_tensor(f(np.linspace(0, win_len - 1, num=seq_len)))

def ZoomOut(img, v):
    "Compresses a sequence on the x-axis"
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = max(lambd, 1 - lambd)
    win_len = int(seq_len * lambd)
    if win_len == seq_len: start = 0
    else: start = np.random.randint(0, seq_len - win_len)
    f = CubicSpline(np.arange(img.shape[-1]), img.cpu(), axis=-1)
    return img.new_tensor(f(np.linspace(0, seq_len - 1, num=win_len)))

def Scale(img, v):
    "Randomly amplifies/ compresses a sequence on the x-axis"
    if v <= 0: return img
    elif np.random.rand() <= .5: return ZoomIn(img, v)
    else: return ZoomOut(img, v)
    
def RandomTimeStep(img, v):
    "Compresses a sequence on the x-axis by randomly selecting sequence steps"
    seq_len = img.shape[-1]
    new_seq_len = int(seq_len * max(.5, (1 - np.random.rand() * v)))
    timesteps = np.sort(np.random.choice(np.arange(seq_len),new_seq_len, replace=False))
    f = CubicSpline(np.arange(len(timesteps)), img[..., timesteps].cpu(), axis=-1)
    output = img.new_tensor(f(np.linspace(0, new_seq_len - 1, num=seq_len)))
    return output

def Blur(img, v):
    "Blurs a sequence applying a filter of type [1, 0..., 1]"
    if v == 3:  filterargs = np.array([1, 0, 1])
    else:
        magnitude = tuple((3, 3 + int(v * 4)))
        n_zeros = int(np.random.choice(np.arange(magnitude[0], magnitude[1] + 1, 2))) - 2
        filterargs = np.array([1] + [0] * n_zeros + [1])
    w = filterargs * np.random.rand(len(filterargs))
    w = w / w.sum()
    output = img.new_tensor(convolve1d(o.cpu(), w, mode='nearest'))
    return output

def Smooth(img, v):
    "Smoothens a sequence applying a filter of type [1, 5..., 1]"
    if v == 3:  filterargs = np.array([1, 5, 1])
    else:
        magnitude = tuple((3, 3 + int(v * 4)))
        n_ones = int(np.random.choice(np.arange(magnitude[0], magnitude[1] + 1, 2))) // 2
        filterargs = np.array([1] * n_ones + [5] + [1] * n_ones)
    w = filterargs * np.random.rand(len(filterargs))
    w = w / w.sum()
    output = img.new_tensor(convolve1d(img.cpu(), w, mode='nearest'))
    return output

def Denoise(img, v, thr= None):
    "Denoises a sequence applying a wavelet decomposition method"
    seq_len = img.shape[-1]
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(img.cpu(), wavelet = 'db4' , mode='per')
    if thr is None:
        # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
        # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
        sigma = (1/0.6745) * maddest(coeff[-2])
        # Calculate the univeral threshold
        uthr = sigma * np.sqrt(2*np.log(seq_len))
        coeff[1:] = (pywt.threshold(c, value=uthr, mode='hard') for c in coeff[1:])
    elif thr == 'random': coeff[1:] = (pywt.threshold(c, value=np.random.rand(), mode='hard') for c in coeff[1:])
    else: coeff[1:] = (pywt.threshold(c, value=thr, mode='hard') for c in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    output = img.new_tensor(pywt.waverec(coeff, wavelet = 'db4' , mode='per')[..., :seq_len])
    return output

def RandomNoise(img, v, wavelet='db4', level=2, mode='constant'):
    "Applys random noise using a wavelet decomposition method"
    if v <= 0: return o
    level = 1 if level is None else level
    coeff = pywt.wavedec(img.cpu(), wavelet, mode=mode, level=level)
    coeff[1:] = [c * (1 + 2 * (np.random.rand() - .5) * v) for c in coeff[1:]]
    output = img.new_tensor(pywt.waverec(coeff, wavelet, mode=mode)[..., :img.shape[-1]])
    return output

def LookBack(img, v):
    "Selects a random number of sequence steps starting from the end"
    if v <= 0: return o
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = min(lambd, 1 - lambd)
    lookback_per = int(lambd * seq_len)
    output = img.clone()
    output[..., :lookback_per] = 0
    return output
    
def VarOut(img, v):
    "Set the value of a random number of variables to zero"
    if v <= 0: return o
    in_vars = img.shape[-2]
    if in_vars == 1: return img
    lambd = np.random.beta(v, v)
    lambd = min(lambd, 1 - lambd)
    p = np.arange(in_vars).cumsum()
    p = p/p[-1]
    p = p / p.sum()
    p = p[::-1]
    out_vars = np.random.choice(np.arange(in_vars), int(lambd * in_vars), p=p, replace=False)
    if len(out_vars) == 0:  return o
    output = img.clone()
    output[...,out_vars,:] = 0
    return output

def CutOut(img, v):
    "Sets a random section of the sequence to zero"
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = min(lambd, 1 - lambd)
    win_len = int(seq_len * lambd)
    start = np.random.randint(-win_len + 1, seq_len)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    output = img.clone()
    output[..., start:end] = 0
    return output

def TimeStepOut(img, v):
    "Sets random sequence steps to zero"
    if v <= 0: return img
    magnitude = min(.5, v)
    seq_len = img.shape[-1]
    timesteps = np.sort(np.random.choice(np.arange(seq_len), int(seq_len * magnitude), replace=False))
    output = img.clone()
    output[..., timesteps] = 0
    return output

def Crop(img, v):
    "Crops a section of the sequence of a predefined length"
    magnitude = min(.5, v)
    seq_len = img.shape[-1]
    win_len = int(seq_len * (1 - magnitude))
    start = np.random.randint(0, seq_len - win_len)
    end = start + win_len
    output = torch.zeros_like(img, dtype=img.dtype, device=img.device)
    output[..., start - end :] = img[..., start : end]
    return output

def RandomCrop(img, v):
    "Crops a section of the sequence of a random length"
    if v <= 0: return o
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = max(lambd, 1 - lambd)
    win_len = int(seq_len * lambd)
    if win_len == seq_len: return img
    start = np.random.randint(0, seq_len - win_len)
    output = torch.zeros_like(img, dtype=img.dtype, device=img.device)
    output[..., start : start + win_len] = img[..., start : start + win_len]
    return output

def RandomResizedCrop(img, v):
    "Crops a section of the sequence of a random length"
    if v <= 0: return img
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = max(lambd, 1 - lambd)
    win_len = int(seq_len * lambd)
    if win_len == seq_len: return img
    start = np.random.randint(0, seq_len - win_len)
    f = CubicSpline(np.arange(win_len), img[..., start : start + win_len].cpu(), axis=-1)
    return img.new_tensor(f(np.linspace(0, win_len, num=seq_len)))

def CenterCrop(img, v):
    "Crops a section of the sequence of a random length from the center"
    if v <= 0: return img
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = max(lambd, 1 - lambd)
    win_len = int(seq_len * lambd)
    start = seq_len // 2 - win_len // 2
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    output = torch.zeros_like(img, dtype=img.dtype, device=img.device)
    output[..., start : end] = img[..., start : end]
    return output

def MaskOut(img, v):
    "Set a random number of steps to zero"
    if v <= 0: return img
    seq_len = img.shape[-1]
    mask = torch.rand_like(img) <= v
    output = img.clone()
    output[mask] = 0
    return output

def TranslateX(img, v):
    "Set a random number of steps to zero"
    if v <= 0: return img
    seq_len = img.shape[-1]
    lambd = np.random.beta(v, v)
    lambd = min(lambd, 1 - lambd)
    shift = int(seq_len * lambd * v)
    if shift == 0: return img
    if np.random.rand() < .5: shift = -shift
    new_start = max(0, shift)
    new_end = min(seq_len + shift, seq_len)
    start = max(0, -shift)
    end = min(seq_len - shift, seq_len)
    output = torch.zeros_like(img, dtype=img.dtype, device=img.device)
    output[..., new_start : new_end] = o[..., start : end]
    return output

def Flip(img, v):
    output = torch.flip(img, [-1])
    return output

def RandomFlip(img, v, p=0.5):
    "Flips the sequence along the x-axis"
    if random.random() < p : return img
    output = torch.flip(img, [-1])
    return output

def Shift(img, v):
    "Shifts and splits a sequence"
    pos = np.random.randint(0, img.shape[-1])
    output = torch.cat((img[..., pos:], img[..., :pos]), dim=-1)
    return output

def RandomRotate(img, v):
    "Randomly rotates the sequence along the z-axis"
    if v <= 0: return img
    flat_x = img.view(img.shape[0], -1)
    ran = flat_x.max(dim=-1, keepdim=True).values - flat_x.min(dim=-1, keepdim=True).values
    trend = torch.linspace(0, 1, o.shape[-1], device=img.device) * ran
    t = (1 + v * 2 * (np.random.rand() - .5) * trend)
    t -= t.mean(-1, keepdim=True)
    if img.ndim == 3: t = t.unsqueeze(1)
    output = img + t
    return output

def Neg(img, v):
    "Applies a negative value to the time sequence"
    return -img

# Cell
def RandomNeg(img, v, p=0.5):
    "Randomly applies a negative value to the time sequence"
    if p < random.random(): return img
    return - img

def FreqNoise(img, v, wavelet='db4',level=2, mode='constant'):
    "Applies noise based on a wavelet decomposition"
    if v<= : return img
    seq_len = img.shape[-1]
    level = 1 if level is None else level
    coeff = pywt.wavedec(img.cpu(), wavelet, mode=mode, level=level)
    coeff[1:] = [c + 2 * (np.random.rand() - .5) * v for c in coeff[1:]]
    output = img.new_tensor(pywt.waverec(coeff, wavelet, mode=mode)[..., :seq_len])
    return output

def FreqWarp(img, v, wavelet='db4',level=2, mode='constant'):
    "Applies warp based on a wavelet decomposition"
    if v <= 0: return img
    seq_len = img.shape[-1]
    level = 1 if level is None else level
    new_x = random_cum_noise_generator(img[:img.shape[-1] // 2], magnitude=v)
    coeff = pywt.wavedec(img.cpu(), wavelet, mode=mode, level=level)
    coeff[1:] = [CubicSpline(np.arange(c.shape[-1]), c, axis=-1)(new_x[:c.shape[-1]]) for c in coeff[1:]]
    output = img.new_tensor(pywt.waverec(coeff, wavelet, mode=mode)[..., :seq_len])
    return output

def FreqScale(img, v, wavelet='db4',level=2, mode='constant'):
    "Modifies the scale based on a wavelet decomposition"
    if v <= 0: return img
    seq_len = img.shape[-1]
    level = 1 if level is None else level
    coeff = pywt.wavedec(img.cpu(), wavelet, mode=mode, level=level)
    coeff[1:] = [c * (1 + 2 * (np.random.rand() - .5) * v) for c in coeff[1:]]
    output = img.new_tensor(pywt.waverec(coeff, wavelet, mode=mode)[..., :seq_len])
    return output

def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
    (MagScale, .02, .2),
    (RandomTimeStep, .02, .2),
    (TimeWarp, .02, .2),
    (MagScaleVar, .02, .2),
    (RandomRotate, .1, .5),
    (MagWarp, .02, .2),
    (TimeNoise, .05, .5),
    ]

    return l


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 10]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 10) * float(maxval - minval) + minval
            img = op(img, val)
        return img
