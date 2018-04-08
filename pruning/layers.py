import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import to_var


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
        
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        # print('mask shape: {}'.format(self.mask.data.size()))
        # print('weight shape {}'.format(self.weight.data.size()))
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            # print(self.weight)
            # print(self.mask)
            # print('weight/mask id: {} {}'.format(self.weight.get_device(), mask_var.get_device()))
            weight = self.weight * mask_var
            return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
