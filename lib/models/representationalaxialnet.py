import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import torchvision
from lib.models.quaternionconv.quaternion_layers import QuaternionConv

import matplotlib.pyplot as plt
import numpy
from PIL import Image

__all__ = ['axialRepresentational26s', 'axialRepresentational35s', 'axialRepresentational50s', 'axialRepresentational50m', 'axialRepresentational50l']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RepresentationalAxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(RepresentationalAxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        #print(q.shape,q_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialRepresentationalBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialRepresentationalBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.quaternion = QuaternionConv(width, width , kernel_size=1, stride=1, bias=False)
        self.bnQuat = norm_layer(width) 
        
        self.hight_block = RepresentationalAxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = RepresentationalAxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  #print("I/P: ",x.shape) #[10, 32, 56, 56]
        
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)   

        
        # display tensor
        #a = torch.Tensor(myOut).cuda()
        #to_img(a)
        
        #imgs = torch.randn(64, 21, 21) 
        
        out = self.quaternion(out)
        out = self.bnQuat(out)     #print("After Quat: ",out.shape)  #[10, 64, 56, 56]
        
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)       #print("After Attention: ",out.shape) #[10, 64, 56, 56]
        
        
        
        out = self.conv_up(out)
        out = self.bn2(out)       #print("After Conv2: ",out.shape) #[10, 128, 56, 56]
        
        if self.downsample is not None:            
            identity = self.downsample(x)
        #out += quat
        
        out += identity
        out = self.relu(out)  #print("Before Return: ",out.shape) #[10, 128, 56, 56]
        return out


class AxialRepresentationalNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialRepresentationalNet, self).__init__()
        self.count = 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(128 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=56)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=56,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=28,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=14,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * block.expansion * s), num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialRepresentationalBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        '''
        myOut = x[-1,:,:,:]
        print(myOut.shape)
        #imgs = torchvision.utils.make_grid(myOut.unsqueeze(1))
        #print(imgs.shape) #[64,56,56] to [3, 466, 466] 
        imgs = myOut.permute(1,2,0)
        np_array = imgs.cpu().detach().numpy()
        #np_array = numpy.uint8(imgs.cpu().detach().numpy().transpose(1, 2, 0))
        #np_array = np_array *255
        img = Image.fromarray(np_array,'RGB')
        f_n = 'after1st_conv'+ str(self.count) +'.png'
        self.count = self.count+1
        img.save(f_n)
        img.show()
        '''
        
        # See note [TorchScript super()]
        x = self.conv1(x)
        #print("after Conv1/stem: ",x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print("after Maxpool: ",x.shape)
        
        
        x = self.layer1(x)  
        #print("after Layer-1: ",x.shape)
        
        
        x = self.layer2(x)  
        #print("after Layer-2: ",x.shape)
        x = self.layer3(x)  
        #print("after Layer-3: ",x.shape)
        x = self.layer4(x)  
        #print("after Layer-4: ",x.shape)
        '''
        myOut = x[-1,:,:,:]
        print(myOut.shape)
        imgs = torchvision.utils.make_grid(myOut.unsqueeze(1))
        print(imgs.shape) #[64,56,56] to [3, 466, 466] 
        imgs = imgs.permute(1,2,0)
        np_array = imgs.cpu().detach().numpy()
        #np_array = numpy.uint8(imgs.cpu().detach().numpy().transpose(1, 2, 0))
        #np_array = np_array *255
        img = Image.fromarray(np_array,'RGB')
        f_n = 'after1st_conv'+ str(self.count) +'.png'
        self.count = self.count+1
        img.save(f_n)
        img.show()
        '''
        
        x = self.avgpool(x) 
        #print("after Avg pool: ",x.shape)
        
        x = torch.flatten(x, 1) 
        #print("after flatten: ",x.shape)
        x = self.fc(x)      
        #print("after fc: ",x.shape)
        
        
        return x

    def forward(self, x):
        return self._forward_impl(x)


def axialRepresentational26s(pretrained=False, **kwargs):
    model = AxialRepresentationalNet(AxialRepresentationalBlock, [1, 2, 4, 1], s=0.5, **kwargs)
    return model

def axialRepresentational35s(pretrained=False, **kwargs):
    model = AxialRepresentationalNet(AxialRepresentationalBlock, [2, 3, 4, 2], s=0.5, **kwargs)
    return model

def axialRepresentational50s(pretrained=False, **kwargs):
    model = AxialRepresentationalNet(AxialRepresentationalBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def axialRepresentational50m(pretrained=False, **kwargs):
    model = AxialRepresentationalNet(AxialRepresentationalBlock, [3, 4, 6, 3], s=0.75, **kwargs)
    return model


def axialRepresentational50l(pretrained=False, **kwargs):
    model = AxialRepresentationalNet(AxialRepresentationalBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model