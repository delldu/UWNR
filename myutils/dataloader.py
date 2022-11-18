import os
import sys
from cv2 import resize
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import cv2
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import random
from PIL import Image
from torch.utils.data import DataLoader
#from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from myutils import dcp
random.seed(2)
np.random.seed(2)
from tqdm import tqdm
import csv

import torch.nn as nn
import torch.nn.functional as F

class GaussFilter(nn.Module):
    """
    3x3 Guassian filter
    """
    def __init__(self):
        super(GaussFilter, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)

        # self.conv.bias.data.fill_(0.0)
        self.conv.weight.data.fill_(0.0625)
        self.conv.weight.data[:, :, 0, 1] = 0.125
        self.conv.weight.data[:, :, 1, 0] = 0.125
        self.conv.weight.data[:, :, 1, 2] = 0.125
        self.conv.weight.data[:, :, 2, 1] = 0.125
        self.conv.weight.data[:, :, 1, 1] = 0.25

    def forward(self, x):
        B, C, H, W = x.size()
        x = F.interpolate(x, (H//32, W//32), mode="bilinear", align_corners=True)
        for i in range(3):
            x = self.conv(x)
        x = (x - x.min())/(x.max() - x.min() + 1e-8)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=True)
        return x


class SUID_Dataset(data.Dataset):
    def __init__(self,path,train,size=128,format='.png'):
        super(SUID_Dataset,self).__init__()
        self.size=size
        #print('crop size',size)
        self.train=train
        self.format=format
        self.uw_imgs_dir=os.listdir(os.path.join(path,'SUID_RAW'))
        self.uw_imgs=[os.path.join(path,'SUID_RAW',img) for img in self.uw_imgs_dir]
        print('Total Images===>',len(self.uw_imgs))
        self.gt_dir=os.path.join(path,'SUID_GT')


    def __getitem__(self, index):
        uw=Image.open(self.uw_imgs[index])
        if isinstance(self.size,int):
            while uw.size[0]< self.size or uw.size[1] < self.size:
                index=random.randint(0,len(self.uw_imgs))
                try:
                    uw=Image.open(self.uw_imgs[index])
                except IndexError:
                    print(index)
        img=self.uw_imgs[index]
        name_syn=img.split('/')[-1]
        id = name_syn
        gt_name=id
        id = name_syn.split('.')[0]
        gt=Image.open(os.path.join(self.gt_dir,gt_name))
        gt=tfs.CenterCrop(uw.size[::-1])(gt)

        if not isinstance(self.size,str):
            if self.train:
                i,j,h,w=tfs.RandomCrop.get_params(uw,output_size=(self.size,self.size))
                uw=FF.crop(uw,i,j,h,w)
                gt=FF.crop(gt,i,j,h,w)

            data,target=self.augData(uw.convert("RGB") ,gt.convert("RGB"))
        return data,target,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)

            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target = FF.rotate(target,90*rand_rot)

        # if self.train == "test":
        #     return data,data,data,data
        

        data=tfs.ToTensor()(data) 



        target = tfs.ToTensor()(target)
        return data,target
    def __len__(self):
        return len(self.uw_imgs)


class UWS_Dataset_Retinex(data.Dataset):
    def __init__(self,path,train,size=128,format='.png',dcp=False):
        super(UWS_Dataset_Retinex,self).__init__()
        self.size=size
        #print('crop size',size)
        self.dcp = dcp
        self.train=train
        self.format=format
        self.uw_imgs_dir=os.listdir(os.path.join(path,'qingxi'))
        self.uw_imgs_dir.sort()
        self.uw_imgs=[os.path.join(path,'qingxi',img) for img in self.uw_imgs_dir][:500]
        print('Total Images===>',len(self.uw_imgs))
        self.gt_dir=os.path.join(path,'GT')
        self.depth_map_dir = os.path.join(path,'DepthMap_size400')

    def __getitem__(self, index):
        uw=Image.open(self.uw_imgs[index])
        
        
        while uw.size[0]< self.size or uw.size[1] < self.size:
            if isinstance(self.size,int):
                index=random.randint(0,len(self.uw_imgs))
                try:
                    uw=Image.open(self.uw_imgs[index])
                    uw=tfs.Resize([self.size,self.size])(uw)
                except IndexError:
                    print(index)
        #index = random.randint(0,499)
        img=self.uw_imgs[index]
        name_syn=img.split('/')[-1].split('_',2)[2]
        id = name_syn
        gt_name=id
        gt=Image.open(os.path.join(self.gt_dir,gt_name))
        #gt=tfs.Resize([400,400])(gt)
        gt=tfs.CenterCrop(uw.size[::-1])(gt)

        depth_map = Image.open(os.path.join(self.depth_map_dir,gt_name).replace('.png','.png'))

        if not isinstance(self.size,str):
            if self.train:
                i,j,h,w=tfs.RandomCrop.get_params(uw,output_size=(self.size,self.size))
                uw=FF.crop(uw,i,j,h,w)
                gt=FF.crop(gt,i,j,h,w)
                depth_map = FF.crop(depth_map,i,j,h,w)
            data,target,depth_map,A_map=self.augData(uw.convert("RGB") ,gt.convert("RGB"),depth_map.convert("L"),index)
        return data,target,depth_map,A_map
    def augData(self,data,target,depth_map,index):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot = random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            depth_map = tfs.RandomHorizontalFlip(rand_hor)(depth_map)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                depth_map=FF.rotate(depth_map,90*rand_rot)
            # data=tfs.RandomRotation((-180,180))(data)
            # target=tfs.RandomRotation((-180,180))(target)
            # depth_map=tfs.RandomRotation((-180,180))(depth_map)
        if self.dcp==True:
            with open('/mnt/data/csx/Documents/underwater_generation2022/A_dcp_train.csv','r',encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
            data=tfs.ToTensor()(data) 
            A_map = torch.ones_like(data)
            A_map[0] = A_map[0]*int(rows[index][2])/255
            A_map[1] = A_map[1]*int(rows[index][1])/255
            A_map[2] = A_map[2]*int(rows[index][0])/255
        else:
            A_map =dcp.MutiScaleLuminanceEstimation(np.uint8(np.array(data)))
            A_map = tfs.ToTensor()(np.float32(A_map))/255
            data=tfs.ToTensor()(data) 

            #A_map = A_map.view_as(data)
        depth_map = np.float32(np.array(depth_map))/255
        
        # ones_matrix = np.ones_like(data.numpy())
        
        depth_map = tfs.ToTensor()(depth_map)
        #print('1',target.shape)
        target = tfs.ToTensor()(target)
        #print('2',target.shape)

        return data,target,depth_map,A_map
    def __len__(self):
        return len(self.uw_imgs)
# UW_train_loader = DataLoader(dataset=UWS_Dataset_Retinex(r'/mnt/data/csx/Documents/dataset/UIEB/',train=True,size=256,dcp=True),shuffle=True,batch_size=18,num_workers=4)
# loop = tqdm(enumerate(UW_train_loader),total=len(UW_train_loader))
# for i ,(a,b,c,d) in loop:
#     print(a)



class UWS_Dataset_Retinex_test(data.Dataset):
    def __init__(self,path,train,size=128,format='.png',dcp=False):
        super(UWS_Dataset_Retinex_test,self).__init__()
        self.size=size
        #print('crop size',size)
        self.dcp = dcp
        self.train=train
        self.format=format
        # self.uw_imgs_dir=os.listdir(os.path.join(path,'qingxi'))
        self.uw_imgs_dir=os.listdir(os.path.join(path))
        random.shuffle(self.uw_imgs_dir)
        #self.uw_imgs_dir.sort()
        # self.uw_imgs=[os.path.join(path,'qingxi',img) for img in self.uw_imgs_dir]
        self.uw_imgs=[os.path.join(path,img) for img in self.uw_imgs_dir]

        self.gauss_filter = GaussFilter()

        print('Total Images===>',len(self.uw_imgs))

    def __getitem__(self, index):
        uw=Image.open(self.uw_imgs[index])
        uw=tfs.Resize([self.size,self.size])(uw)

        A_map,data=self.augData(uw.convert("RGB"),index)
        return A_map,data#,target,depth_map,A_map,id
    def augData(self,data,index):#,target,depth_map):

        if self.dcp==True:
            with open('/mnt/data/csx/Documents/underwater_generation2022/A_dcp.csv','r',encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
            data=tfs.ToTensor()(data) 
            A_map = torch.ones_like(data)
            A_map[0] = A_map[0]*int(rows[index][2])/255
            A_map[1] = A_map[1]*int(rows[index][1])/255
            A_map[2] = A_map[2]*int(rows[index][0])/255
        else:
            # A_map =dcp.MutiScaleLuminanceEstimation(np.uint8(np.array(data)))
            # A_map = tfs.ToTensor()(np.float32(A_map))/255

            #A_map = A_map.view_as(data)
            data=tfs.ToTensor()(data)
            A_map = self.gauss_filter(data.unsqueeze(0)).squeeze(0) 

            print("data.size(): ", data.size(), data.mean())

        return A_map,data# data,target,depth_map,
    def __len__(self):
        return len(self.uw_imgs)




# data = UWS_discriminator_dataloader(r'/mnt/data/yt/Documents/Zero-Reference-Underwater-Image-Enhancedment/UIEB')
# for img,label in data:
#     print(img,label)

			


