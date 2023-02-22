from model_utils import conv_block,focus_downsample_2x,residual_block,residual_se_block,shuffle_residual_block
import torch
import torch.nn as nn

class Conventional_CNN(nn.Module):
    def __init__(self,out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10):
        super(Conventional_CNN,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample1 = conv_block(3,out_channels[0],ks=7,stride=2)
        self.stage1 = self._make_stage(conv_block,stage_layer_num[0],in_c=out_channels[0],out_c=out_channels[0])
        self.downsample2 = conv_block(out_channels[0],out_channels[1],ks=3,stride=2)
        self.stage2 = self._make_stage(conv_block,stage_layer_num[1],in_c=out_channels[1],out_c=out_channels[1])
        self.downsample3 = conv_block(out_channels[1],out_channels[2],ks=3,stride=2)
        self.stage3 = self._make_stage(conv_block,stage_layer_num[2],in_c=out_channels[2],out_c=out_channels[2])
        self.downsample4 = conv_block(out_channels[2],out_channels[3],ks=3,stride=2)
        self.stage4 = self._make_stage(conv_block,stage_layer_num[3],in_c=out_channels[3],out_c = out_channels[3])

        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels[-1],num_class)
        
        
    def _make_stage(self,block,layer,**kwargs):
        module_list = nn.Sequential()
        for i in range(layer):
            module_list.add_module(f'layer{i}',block(**kwargs))
            module_list.add_module(f'layer{i}_relu',nn.ReLU(inplace=True))
        return module_list
    
    def forward(self,x):
        bs,_,_,_ = x.shape
        x = self.stage1(self.downsample1(x))
        x = self.stage2(self.downsample2(x))
        x = self.stage3(self.downsample3(x))
        x = self.stage4(self.downsample4(x))
        x = self.fc(self.GAP(x).view(bs,-1))

        return x

class Residual_CNN(Conventional_CNN):
    def __init__(self,out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10):
        super(Residual_CNN,self).__init__(out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10)
        self.stage1 = self._make_stage(residual_block,stage_layer_num[0],in_c=out_channels[0],out_c=out_channels[0])
        self.stage2 = self._make_stage(residual_block,stage_layer_num[1],in_c=out_channels[1],out_c=out_channels[1])
        self.stage3 = self._make_stage(residual_block,stage_layer_num[2],in_c=out_channels[2],out_c=out_channels[2])
        self.stage4 = self._make_stage(residual_block,stage_layer_num[3],in_c=out_channels[3],out_c = out_channels[3])

class SE_Residual_CNN(Conventional_CNN):
    def __init__(self,out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10):
        super(SE_Residual_CNN,self).__init__(out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10)
        self.stage1 = self._make_stage(residual_se_block,stage_layer_num[0],in_c=out_channels[0],out_c=out_channels[0])
        self.stage2 = self._make_stage(residual_se_block,stage_layer_num[1],in_c=out_channels[1],out_c=out_channels[1])
        self.stage3 = self._make_stage(residual_se_block,stage_layer_num[2],in_c=out_channels[2],out_c=out_channels[2])
        self.stage4 = self._make_stage(residual_se_block,stage_layer_num[3],in_c=out_channels[3],out_c = out_channels[3])

class Shuffle_Residual_CNN(Conventional_CNN):
    def __init__(self,out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10,groups=[4,8,16,16]):
        super(Shuffle_Residual_CNN,self).__init__(out_channels=[64,256,512,512],stage_layer_num=[1,2,3,3],num_class=10)
        self.stage1 = self._make_stage(shuffle_residual_block,stage_layer_num[0],in_c=out_channels[0],out_c1=out_channels[0],out_c2=out_channels[0],groups=groups[0])
        self.stage2 = self._make_stage(shuffle_residual_block,stage_layer_num[1],in_c=out_channels[1],out_c1=out_channels[1],out_c2=out_channels[1],groups=groups[1])
        self.stage3 = self._make_stage(shuffle_residual_block,stage_layer_num[2],in_c=out_channels[2],out_c1=out_channels[2],out_c2=out_channels[2],groups=groups[2])
        self.stage4 = self._make_stage(shuffle_residual_block,stage_layer_num[3],in_c=out_channels[3],out_c1=out_channels[3],out_c2=out_channels[3],groups=groups[3])
if __name__ == "__main__":
    input = torch.zeros((2,3,128,128))
    model1 = Conventional_CNN()
    model2 = Residual_CNN()
    model3 = SE_Residual_CNN()
    model4 = Shuffle_Residual_CNN()
    print(model4)
    print(model4(input).size())

    def num_param(model):
        return sum([param.numel() for param in model.parameters()])
    
    print(num_param(model1)/1e6)
    print(num_param(model2)/1e6)
    print(num_param(model3)/1e6)
    print(num_param(model4)/1e6)
