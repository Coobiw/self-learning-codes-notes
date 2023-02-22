import torch
from torchvision import datasets as tvd
from torch.utils.data import DataLoader,Dataset
from augmentation import RandomCrop
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

SEED = 50
random.seed(SEED)

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def voc_label_indices(colormap):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

DOWNLOAD = False
voc_seg_2012_train = tvd.VOCSegmentation(root='/media/charon/ubuntu_data/seg_dataset/voc2012_seg',image_set='train',
                            download=DOWNLOAD)
voc_seg_2012_val = tvd.VOCSegmentation(root='/media/charon/ubuntu_data/seg_dataset/voc2012_seg',image_set='val',
                            download=DOWNLOAD)


class PSP_Dataset(Dataset):
    def __init__(self,dataset,n_classes,crop=(320,480)) -> None: # crop:(h,w) n_classes : exclude the background
        super().__init__()
        self.dataset = dataset
        self.n_cls = n_classes
        self.size = crop
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        if self.size != None:
            tag = True
            while tag:
                img,mask = self.dataset[index]
                w,h = img.size
                th,tw = self.size
                if w < tw or h < tw:
                    index = random.randint(0,len(self.dataset)-1)
                else:
                    tag = False

            random_crop = RandomCrop(size=self.size)
            img,mask = random_crop((img,mask))
        else:
            img,mask = self.dataset[index]
        mask = mask.convert('RGB')
        mask = voc_label_indices(mask)
        
        img = transforms.ToTensor()(img)
        _,h,w = img.shape
        y_cls = torch.zeros((self.n_cls))
        for ih in range(h):
            for iw in range(w):
                if mask[ih,iw] != 0:
                    y_cls[int(mask[ih,iw])-1] = 1
        return img,mask,y_cls

TRAINSET = PSP_Dataset(voc_seg_2012_train,20,crop=(320,480))
VALSET = PSP_Dataset(voc_seg_2012_val,20,crop=None)