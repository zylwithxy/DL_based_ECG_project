import torch
from torch import nn
import numpy as np
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se=nn.Sequential(
            nn.Conv1d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv1d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv1d(2,1,kernel_size=kernel_size,padding=kernel_size//2) # 保证填充之后维度没有改变
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=-2,keepdim=True) # _ 返回的是最大元素的下标
        avg_result=torch.mean(x,dim=-2,keepdim=True)
        result=torch.cat([max_result,avg_result], dim = -2)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out') # 'fan_out' preserves the magnitudes in the backwards pass
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # b, c, _, _ = x.size()
        c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual