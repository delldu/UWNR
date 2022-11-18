import numpy as np
import cv2
import pdb

def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    #image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)



def estimation_atmosphere(image,sigmaX = 10,blocksize=61):
    backscattering_light = cv2.GaussianBlur(image,(blocksize,blocksize),sigmaX)
    return backscattering_light


def kernel_size(sigma):
    return 3.0 + 2.0*(sigma - 0.8)/0.3

import math
import torch

r = 0
s = [15,60,90]
class MyGaussianBlur(torch.nn.Module):
    #初始化
    def __init__(self, radius=1, sigema=1.5):
        super(MyGaussianBlur,self).__init__()
        self.radius=radius
        self.sigema=sigema
    #高斯的计算公式
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
 
    #滤波模板
    def template(self):
        sideLength=self.radius*2+1
        result=np.zeros((sideLength, sideLength))
        for i in range(0, sideLength):
            for j in range(0,sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all= result.sum()
        return result/all
    #滤波函数
    def filter(self, image, template):
        kernel = torch.FloatTensor(template).cuda()
        kernel2 = kernel.expand(3, 1, 2*r+1, 2*r+1)
        weight = torch.nn.Parameter(data=kernel2, requires_grad=False)
        new_pic2 = torch.nn.functional.conv2d(image, weight, padding=r, groups=3)
        return new_pic2

# print(loss.item())
def MutiScaleLuminanceEstimation1(img):
    guas_15 = MyGaussianBlur(radius=r, sigema=15).cuda()
    temp_15 = guas_15.template()
        
    guas_60 = MyGaussianBlur(radius=r, sigema=60).cuda()
    temp_60 = guas_60.template()

    guas_90 = MyGaussianBlur(radius=r, sigema=90).cuda()
    temp_90 = guas_90.template()
    x_15 = guas_15.filter(img, temp_15)
    x_60 = guas_60.filter(img, temp_60)
    x_90 = guas_90.filter(img, temp_90)
    img = (x_15+x_60+x_90)/3

    return img


def MutiScaleLuminanceEstimation(img):
    #  img.shape -- (400, 400, 3), dtype=uint8, 0-255

    sigma_list  = [15,60,90]
    w,h,c = img.shape
    img = cv2.resize(img,dsize=None,fx=0.3,fy=0.3) # (120, 120, 3)

    # kernel_size(15)

    Luminance = np.ones_like(img).astype(np.float)
    for sigma in sigma_list:
        kernel = 6 * sigma + 1
        Luminance1 = np.log10(cv2.GaussianBlur(img, (kernel, kernel), sigma))
        Luminance1 = np.clip(Luminance1,0,255)
        Luminance += Luminance1
    Luminance =  Luminance/3
    L = (Luminance - np.min(Luminance)) / (np.max(Luminance) - np.min(Luminance)+0.0001)
    L =  np.uint8(L*255)
    L =  cv2.resize(L,dsize=(h,w))
    #  L.shape -- (400, 400, 3), dtype=uint8

    return L

