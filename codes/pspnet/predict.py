import torch
from dataloader import TRAINSET,VALSET,VOC_CLASSES,VOC_COLORMAP
from pspnet import PSPNet
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import random
import time
import torch.nn as nn

import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import List,Optional,Union,Iterable
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('seed',type=int)
parser.add_argument('--model-path',type=str,required=True)
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.makedirs(f"./seed{SEED}",exist_ok=True)

class Seg_predictor:
    def __init__(self,model,stride,classes,colormap):
        self.stride = stride
        self.classes = classes
        self.colormap = colormap
        self.model = model
    
    @staticmethod
    def read_img(img_path):
        img = Image.open(img_path).convert('RGB')
        transform = transforms.ToTensor()
        img = transform(img)
        img = img.unsqueeze(dim=0)
        return img 

    def get_resize_shape(self,img:torch.Tensor):
        _,_,H,W = img.shape
        if H//self.stride:
            new_h = H - (H%self.stride)
        else:
            new_h = H
        if W//self.stride:
            new_w = W - (W%self.stride)
        else:
            new_w = W
        return new_h,new_w

    def predict(self,img,device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if isinstance(img,torch.Tensor):
            img = img.to(device)
        elif isinstance(img,np.ndarray):
            img = torch.from_numpy(img)
            img = img.to(device)
        else:
            raise TypeError
        
        assert img.ndim >= 3,'Only adaptive for RGB image'
        if img.ndim == 3:
            img = img.unsqueeze(dim=0)

        _,_,h,w = img.shape
        new_h,new_w = self.get_resize_shape(img)
        resize_transform = transforms.Resize(size=(new_h,new_w))
        mask_interpolation_transform = transforms.Resize(size=(h,w),interpolation=transforms.InterpolationMode.NEAREST)
        
        input_img = resize_transform(img)
        self.model = self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            prob_mask,_ = self.model(input_img)
            cls_mask = prob_mask.argmax(dim=1) # cls_mask: [B,H,W]
            mask_out = self.cls2color(cls_mask)
            mask_out = mask_interpolation_transform(mask_out)

        # b,h,w = cls_mask.shape
        # for ib in range(b):
        #     for ih in range(h):
        #         print('\n')
        #         for iw in range(w):
        #             print(cls_mask[ib,ih,iw].item(),end=' ')
        return mask_out
    
    def cls2color(self,cls_mask):
        cmap = torch.tensor(self.colormap).to(cls_mask.device)
        mask_out = cmap[cls_mask.long()].permute(0,3,1,2).contiguous()
        return mask_out

    def mask_visualize(self,mask:torch.Tensor,plot:bool=True,plot_table:bool=False):
        if not isinstance(mask,torch.Tensor):
            raise TypeError
        assert mask.ndim == 4,'only process tensor with 4 dimensions'
        assert mask.shape[0]== 1,'only process the tensor whose batch_size is equal to 1'

        mask = mask.squeeze(dim=0)
        mask = mask.permute(1,2,0).contiguous()
        mask = mask.cpu().numpy().clip(min=0,max=255).astype('uint8')
        h,w,_ = mask.shape

        if plot:
            if plot_table:
                fig,ax = self.plot_colortable(mask,(h,w))
            else:
                fig = plt.figure()
                ax = plt.gca()
                ax.set_xlim(0,w)
                ax.set_ylim(h,0)
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                ax.set_axis_off()
                plt.imshow(mask,vmin=0,vmax=255)
            img_name = time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.png'
            plt.savefig(f'./seed{SEED}/'+img_name)
            del fig,ax
            # cv2.imwrite(f'./seed{SEED}/p'+img_name,cv2.cvtColor(mask,cv2.COLOR_RGB2BGR))
    
    def plot_colortable(self,mask,img_hw=None,emptycols=0):
        cell_width = 212
        cell_height = 22
        swatch_width = 48
        margin = 12
        topmargin = 40
        h_margin = 5

        n_cls = len(self.classes)
        ncols = 2 - emptycols
        nrows = n_cls//ncols + int(n_cls%ncols > 0)
        img_h,img_w = (0,0) if img_hw is None else img_hw
        width = cell_width * 2 + 2 * margin + img_w
        height = max(img_h,cell_height * nrows) + margin + topmargin 
        dpi = 72

        color_cls_dict = dict()
        for i in range(n_cls):
            color_cls_dict[self.classes[i]] = self.colormap[i]
        
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = plt.gca()
        ax.set_xlim(0,img_w + margin + cell_width * 2)
        ax.set_ylim(img_h,0)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_off()

        ax.imshow(mask,vmin=0,vmax=255)
        for i,item in enumerate(color_cls_dict.items()):
            cls,color = item
            c_lst = [None,None,None]
            for j in range(3):
                c_lst[j] = color[j] / 255

            row = i % nrows
            col = i // nrows
            y = 20 + row * (cell_height + h_margin)

            swatch_start_x = cell_width * col + img_w + 10
            swatch_end_x = cell_width * col + swatch_width + img_w + 10
            text_pos_x = cell_width * col + swatch_width + 7 + img_w + 10

            ax.text(text_pos_x, y, cls, fontsize=14,
                    horizontalalignment='left',
                    verticalalignment='center')

            ax.hlines(y, swatch_start_x, swatch_end_x,
                    color=c_lst, linewidth=18)

        return fig,ax
        
        

if __name__ == "__main__":
    dataset = TRAINSET
    valset_len = len(dataset)
    index = random.randint(0,valset_len-1)
    img,gt_mask,y_cls = dataset[index]
    # print(y_cls)
    img = img.unsqueeze(dim=0)
    gt_mask = gt_mask.unsqueeze(dim=0)

    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34').cpu()
    model = nn.DataParallel(model)
    with open(args.model_path,'rb') as f:
        model.load_state_dict(torch.load(f,map_location='cpu'))

    predictor = Seg_predictor(model=model,stride=32,classes=VOC_CLASSES,colormap=VOC_COLORMAP)
    
    # predictor.plot_colortable()
    # plt.savefig('./test.png')

    predictor.mask_visualize(img*255,plot_table=False)
    time.sleep(2)
    predictor.mask_visualize(predictor.cls2color(gt_mask),plot_table=True)
    time.sleep(2)
    predictor.mask_visualize(predictor.predict(img),plot_table=True)
    time.sleep(2)
    predictor.mask_visualize(img*255 + predictor.cls2color(gt_mask),plot_table=True)
    time.sleep(2)
    predictor.mask_visualize(img*255 + predictor.predict(img).to(img.device),plot_table=True)


