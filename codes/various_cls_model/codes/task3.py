import torch
import torch.optim as optim
import torch.nn as nn
import os
from dataloader import get_loader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import Conventional_CNN,Residual_CNN,SE_Residual_CNN,Shuffle_Residual_CNN

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SEED = 729608
# torch random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# os python hash seed, make experiment reproducable
os.environ['PYTHONHASHSEED'] = str(SEED)
# gpu algorithom 
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Generator SEED
Generator = torch.Generator()
Generator.manual_seed(SEED)

class AverageMeter:
    def __init__(self):
        self.sum = 0.
        self.n = 0
        self.avg = 0.
    
    def update(self,val,n,multiply=True):
        self.n += n
        if multiply:
            self.sum += val * n
        else:
            self.sum += val
        self.avg = self.sum / self.n

def train(model,fig_name:str):
    train_loader,test_loader = get_loader(normalize_tag=True,bs=128)
    lr = 1e-3
    epochs = 30
    weight_decay = 1e-5

    weight_params = []
    bias_params = []
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.Linear,nn.BatchNorm2d)):
            weight_params.append(module.weight)
            if not (module.bias is None):
                bias_params.append(module.bias)

    optimizer = optim.Adam([{'params':weight_params},{'params':bias_params,'weight_decay':0.}],lr=lr,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    model = model.to(DEVICE)

    pbar = tqdm(range(epochs))
    train_loss_lst = []
    test_loss_lst = []
    train_acc_lst = []
    test_acc_lst = []
    best_acc = 0.

    for i in pbar:
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        test_loss = AverageMeter()
        test_acc =AverageMeter()
        for data,label in train_loader:
            model.train()
            data,label = data.to(DEVICE),label.to(DEVICE)
            pred = model(data)
            loss = criterion(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(),data.shape[0])

            pred = pred.max(dim=1)[1]
            acc_num = (pred==label).sum()
            train_acc.update(int(acc_num.detach()),data.shape[0],multiply=False)
        
        for data,label in test_loader:
            model.eval()
            with torch.no_grad():
                data,label = data.to(DEVICE),label.to(DEVICE)
                pred = model(data)
                loss = criterion(pred,label)
                test_loss.update(loss.item(),data.shape[0])

                pred = pred.max(dim=1)[1]
                acc_num = (pred==label).sum()
                test_acc.update(int(acc_num.detach()),data.shape[0],multiply=False)
        
        log_str = f'Epoch {i+1}: Train Loss : {train_loss.avg :.3f}  |  Train Acc : {train_acc.avg * 100:.2f}  |  Test Loss : {test_loss.avg :.3f}  |  Test Acc : {test_acc.avg * 100:.2f}'
        pbar.set_description(log_str)
        train_loss_lst.append(train_loss.avg)
        train_acc_lst.append(train_acc.avg * 100)
        test_loss_lst.append(test_loss.avg)
        test_acc_lst.append(test_acc.avg * 100)
        best_acc = max(best_acc,test_acc.avg*100)

    plt.figure(figsize = (12,10))
    plt.plot(train_loss_lst,label='train')
    plt.plot(test_loss_lst,label = 'test')
    plt.legend(loc='best')
    plt.savefig(f'./{fig_name}_loss.png')

    plt.figure(figsize = (12,10))
    plt.plot(train_acc_lst,label='train')
    plt.plot(test_acc_lst,label = 'test')
    plt.legend(loc='best')
    plt.savefig(f'./{fig_name}_acc.png')

    return best_acc
def num_param(model):
        return sum([param.numel() for param in model.parameters()])

def main():
    model_lst = [Conventional_CNN(),Residual_CNN(),SE_Residual_CNN(),Shuffle_Residual_CNN()]
    name_lst = ['conventional','residual','se_residual','shuffle_residual']
    acc_lst = []
    num_param_lst = []

    for i in range(len(name_lst)):
        acc_lst.append(train(model_lst[i],name_lst[i]))
    
    for model in model_lst:
        num_param_lst.append(round(num_param(model)/1e6))

    color_lst = ['red','pink','blue','green']
    print(acc_lst)
    plt.figure(figsize = (12,10))
    for i in range(4):
        plt.scatter(num_param_lst[i],acc_lst[i],s=500,color=color_lst[i],alpha=0.5,label=name_lst[i])
    plt.rcParams["legend.markerscale"] = 0.5
    plt.legend(loc='best')
    # plt.xticks(num_param_lst,[f'{num_params:.2f}' for num_params in num_param_lst])
    plt.xticks(num_param_lst,num_param_lst)
    plt.xlabel('num of parameters(M)',fontsize=15)
    plt.ylabel('best accuracy(%)',fontsize=15)
    plt.savefig('./num_para_acc.png')

if __name__ == "__main__":
    main()

    


