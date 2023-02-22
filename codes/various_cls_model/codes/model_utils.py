import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_c,out_c,ks=3,stride=1): # conventional conv+bn block
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=ks,stride=stride,padding=ks//2,bias=False),
        nn.BatchNorm2d(out_c),
    )

def depthwise_conv_block(in_c,out_c,ks=3): # mobilenet core operation
    assert out_c % in_c == 0
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c,out_channels=out_c,groups=in_c,kernel_size=ks,stride=1,padding=ks//2,bias=False),
        nn.BatchNorm2d(out_c)
    )

def channel_shuffle(x,groups): # shufflenet core operation
    bs,C,H,W = x.shape
    assert C % groups == 0
    c_per_group = C // groups
    x = x.view(bs,groups,c_per_group,H,W)
    x = x.permute(0,2,1,3,4).contiguous()
    x = x.view(bs,C,H,W)
    return x

def group_conv_block(in_c,out_c,groups,ks=3,stride=1):
    assert in_c % groups ==0 and out_c % groups ==0
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c,out_channels=out_c,groups=groups,kernel_size=ks,stride=stride,padding=ks//2,bias=False),
        nn.BatchNorm2d(out_c)
    )

class shuffle_residual_block(nn.Module):
    def __init__(self,in_c,out_c1,out_c2,groups,ks=3):
        super(shuffle_residual_block,self).__init__()
        self.gconv1 = group_conv_block(in_c,out_c1,groups=groups,ks=1)
        self.dwconv = depthwise_conv_block(out_c1,out_c1,ks=3)
        self.gconv2 = group_conv_block(out_c1,out_c2,groups=groups,ks=1)

        self.relu = nn.ReLU(inplace=False)
        self.residual = nn.Conv2d(in_c,out_c2,1,1,0) if in_c != out_c2 else nn.Identity()
        self.groups = groups
    
    def forward(self,x):
        residual = self.residual(x)

        out = self.relu(self.gconv1(x))
        out = channel_shuffle(x,groups=self.groups)
        out = self.gconv2(self.dwconv(out))

        out += residual
        out = self.relu(out)

        return out

class focus_downsample_2x(nn.Module): # focus module to downsample proposed in YOLOv5
    def __init__(self,in_c):
        super(focus_downsample_2x,self).__init__()
        self.conv = nn.Conv2d(4*in_c,in_c,1,1,0)

    def forward(self,x):
        bs,C,H,W = x.shape
        assert H//2 == 0 and W//2 == 0
        return self.conv(torch.cat((x[:,:,::2,::2],x[:,:,::2,1::2],x[:,:,1::2,::2],x[:,:,1::2,1::2]),dim=1))

class residual_block(nn.Module): # residual block proposed in resnet
    def __init__(self,in_c,out_c,ks=3):
        super(residual_block,self).__init__()
        self.conv1 = conv_block(in_c,out_c,ks,stride=1)
        self.conv2 = conv_block(out_c,out_c,ks,stride=1)
        self.relu = nn.ReLU(inplace=False)
        self.residual = nn.Conv2d(in_c,out_c,1,1,0) if in_c != out_c else nn.Identity()
    
    def forward(self,x):
        residual = self.residual(x)

        out = self.relu(self.conv1(x))
        out = self.conv2(x)

        out += residual
        out = self.relu(out)

        return out

class residual_se_block(nn.Module): # se block(proposed in SENet) + residual block 
    def __init__(self,in_c,out_c,ks=3):
        super(residual_se_block,self).__init__()
        self.conv1 = conv_block(in_c,out_c,ks,stride=1)
        self.conv2 = conv_block(out_c,out_c,ks,stride=1)
        self.relu = nn.ReLU(inplace=False)
        self.residual = nn.Conv2d(in_c,out_c,1,1,0) if in_c != out_c else nn.Identity()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(out_c,out_c//16)
        self.fc2 = nn.Linear(out_c//16,out_c)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        bs,_,_,_ = x.shape
        residual = self.residual(x)

        out = self.relu(self.conv1(x))
        out = self.conv2(x)

        attn = out
        attn = self.GAP(attn).view(bs,-1)
        attn = self.relu(self.fc1(attn))
        attn = self.sigmoid(self.fc2(attn))

        out = out*attn.view(bs,-1,1,1) + residual
        out = self.relu(out)

        return out

