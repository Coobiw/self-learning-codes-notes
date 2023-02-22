from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transform
import os

TRAIN_TRANSFORM = transform.Compose([transform.ToTensor(),transform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),transform.Resize(size=[128,128])])
TEST_TRANSFORM = transform.Compose([transform.ToTensor(),transform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),transform.Resize(size=[128,128])])
DATA_ROOT_DIR = './data/'

DOWNLOAD_TAG  = False
if not os.path.exists(DATA_ROOT_DIR):
    os.makedirs(DATA_ROOT_DIR)
    DOWNLOAD_TAG = True

def get_loader(normalize_tag=True,bs=128):
    assert isinstance(bs,int) or bs == 'full'
    CIFAR10_trainset = CIFAR10(root=DATA_ROOT_DIR,train=True,transform=TRAIN_TRANSFORM if normalize_tag else transform.ToTensor(),download=DOWNLOAD_TAG)
    CIFAR10_testset = CIFAR10(root=DATA_ROOT_DIR,train=False,transform=TEST_TRANSFORM if normalize_tag else transform.ToTensor(),download=DOWNLOAD_TAG)
    return DataLoader(dataset=CIFAR10_trainset,batch_size=len(CIFAR10_trainset) if bs=='full' else bs,shuffle=True,num_workers=2),\
        DataLoader(dataset=CIFAR10_testset,batch_size=len(CIFAR10_testset) if bs=='full' else bs,shuffle=False,num_workers=2)

if __name__ == "__main__":
    get_loader()