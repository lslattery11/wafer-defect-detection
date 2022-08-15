import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from collections import OrderedDict
from itertools import chain

class CNN_SPP_Net(nn.Module):
    def __init__(
        self,
        cnn_channels: Tuple[int],
        spp_output_sizes: List[Tuple[int]],
        linear_output_sizes: Tuple[int],
        ):
        super().__init__()

        self.cnn_channels = cnn_channels
        self.spp_output_sizes = spp_output_sizes

        first_linear_dim=self.compute_spp_output_size()
        self.linear_dims=(first_linear_dim,)+linear_output_sizes
        self.construct_net()

    def construct_net(self):

        cnn_layer_list=[self.cnn_layer(i) for i in range(len(self.cnn_channels)-1)]
        self.cnn_layers=nn.Sequential(
            OrderedDict(list(chain(*cnn_layer_list)))
        )

        linear_layer_list=[self.linear_layer(i) for i in range(len(self.linear_dims)-1)]
        self.linear_layers=nn.Sequential(
            OrderedDict(list(chain(*linear_layer_list)))
        )

    def cnn_layer(
        self,
        layer_idx: int,
        ):
        num_input_channels,num_output_channels=self.cnn_channels[layer_idx:layer_idx+2]

        return [
            (f'conv2d{layer_idx}',nn.Conv2d(num_input_channels, num_output_channels, kernel_size=3, stride=1, padding='same')),
            (f'bnorm2d{layer_idx}',nn.BatchNorm2d(num_output_channels)),
            (f'cnn-relu{layer_idx}',nn.ReLU()),
            (f'maxpool2d{layer_idx}',nn.MaxPool2d(kernel_size=2, stride=2)),
        ]

    def spp_layer(
        self,
        input_tensor
        ):
        def compute_spp_cat_tensor(input_tensor):
            for i,output_size in enumerate(self.spp_output_sizes):
                output=nn.AdaptiveMaxPool2d(output_size)(input_tensor)
                if i==0:
                    spp_output_tensor=output.view(output.size()[0],-1)
                else:
                    spp_output_tensor=torch.cat((spp_output_tensor,output.view(output.size()[0],-1)),1)
            return spp_output_tensor

        return compute_spp_cat_tensor(input_tensor)
            
    def linear_layer(
        self,
        layer_idx: int,
        ):
        if layer_idx < len(self.linear_dims)-2:
            return [
                (f'linear{layer_idx}',nn.Linear(self.linear_dims[layer_idx],self.linear_dims[layer_idx+1])),
                (f'dropout{layer_idx}',nn.Dropout(p=0.2)),
                (f'linear_relu{layer_idx}',nn.ReLU()),
                ]
        else:
            return [
                    (f'linear{layer_idx}',nn.Linear(self.linear_dims[layer_idx],self.linear_dims[layer_idx+1])),
                    #(f'linear_softmax{layer_idx}',nn.Softmax()),
                ]

    def compute_spp_output_size(self):
        return np.sum(self.cnn_channels[-1]*np.prod(self.spp_output_sizes,axis=1))

    def forward(self, x):

        x=self.cnn_layers(x)
        x=self.spp_layer(x)
        x=self.linear_layers(x)

        return x