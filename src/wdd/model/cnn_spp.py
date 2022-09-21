import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from typing import List, Tuple, TypedDict
from collections import OrderedDict
from itertools import chain
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split

from wdd.data_handling.pull_data import get_processed_data
from wdd.data_handling.torch_dataset import WaferDataset
from wdd.data_handling.augment_data import wafer_train_transforms

class cnn_spp_hypDict(TypedDict):
    """
    dictionary for the hyperparameters of the CNN_SPP_Net model.
    """
    cnn_channels: Tuple[int]
    spp_output_sizes: List[Tuple[int]]
    linear_output_sizes: Tuple[int]

class training_functions_Dict(TypedDict):
    """
    dictionary for the dataloaders and loss functions for training and validating the CNN_SPP_Net model.
    """
    trainingLoader: DataLoader
    validLoader: DataLoader
    trainingLossfn: CrossEntropyLoss
    validLossfn: CrossEntropyLoss

class CNN_SPP_Net(nn.Module):
    """
    incomplete description.
    """
    def __init__(
        self,
        hyperparameters: cnn_spp_hypDict,
        ):
        super().__init__()

        self.cnn_channels = hyperparameters['cnn_channels']
        self.spp_output_sizes = hyperparameters['spp_output_sizes']

        first_linear_dim=self.compute_spp_output_size()
        self.linear_dims=(first_linear_dim,)+hyperparameters['linear_output_sizes']
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
                (f'bnorm1d{layer_idx}',nn.BatchNorm1d(self.linear_dims[layer_idx+1])),
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

    def predict(self,dataloader):
        self.eval()
        device=next(self.parameters()).device
        truths=torch.tensor([],dtype=torch.int64,device=device)
        preds=torch.tensor([],dtype=torch.int64,device=device)
        pred_probs=torch.tensor([],device=device)

        with torch.no_grad():
            for inputs,labels in dataloader:
                inputs,labels=inputs.to(device),labels.to(device)
                x=self.forward(inputs)
                logits=torch.nn.Softmax(dim=1)(x)
                batch_pred_probs,batch_preds=torch.max(logits,dim=1)

                truths=torch.cat((truths,labels),0)
                preds=torch.cat((preds,batch_preds),0)
                pred_probs=torch.cat((pred_probs,batch_pred_probs),0)

        return truths,preds,pred_probs
        
    def init_weights(self):
        """
        initialize weights using xavier_uniform method.
        """
        def xavier_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.10)

        self.apply(xavier_weights)

    def add_training_functions(self, training_funcs : training_functions_Dict):
        """
        add DataLoaders and Lossfn to the class
        """
        self.trainingLoader=training_funcs['trainingLoader']
        self.validLoader=training_funcs['validLoader']
        self.trainingLossfn=training_funcs['trainingLossfn']
        self.validLossfn=training_funcs['validLossfn']

def make_spp_training_net(config):

    model_parameters=config['model_parameters']
    net=CNN_SPP_Net(model_parameters)
    net.init_weights()
    if torch.cuda.is_available() and config['use_cuda']==True:
        device=torch.device('cuda')
        net.to(device)

    #load data. split train_df into training and valid dataframes.
    train_df,_=get_processed_data()
    train_df,valid_df=train_test_split(train_df, test_size=0.2,random_state=42)

    #apply data augmentation to training set.
    transform_prob_threshold=config['transform_prob_threshold']
    training_set=WaferDataset(train_df,transform=wafer_train_transforms(transform_prob_threshold))
    valid_set=WaferDataset(valid_df)

    #calculate class weights for training set. sample with weights.
    training_class_weights=torch.Tensor([1/training_set.len])*torch.Tensor([training_set.y.count(i) for i in range(9)])
    assert(np.isclose(training_class_weights.sum(),1)),'class_weights must sum to be one'

    sample_weights=torch.Tensor([1/training_class_weights[i] for i in training_set.y])

    sampler=WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights))

    #calculate class weights for validation set.
    valid_class_weights=torch.Tensor([1/valid_set.len])*torch.Tensor([valid_set.y.count(i) for i in range(9)])
    assert(np.isclose(valid_class_weights.sum(),1)),'valid_class_weights must sum to be one'

    #create data loaders. use sampler for training_loader
    training_loader = DataLoader(training_set, batch_size=1 , num_workers=0,sampler=sampler)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=0)

 
    #define the loss function with classification Cross-Entropy loss and an optimizer with Adam optimizer. Reweight the validation loss by validation set class weights.
    train_loss_fn = CrossEntropyLoss()
    valid_loss_fn = CrossEntropyLoss(weight=valid_class_weights.reciprocal())

    training_funcs=dict(
        trainingLoader=training_loader,
        validLoader=valid_loader,
        trainingLossfn=train_loss_fn,
        validLossfn=valid_loss_fn,
    )
    net.add_training_functions(training_funcs)

    return net

    
