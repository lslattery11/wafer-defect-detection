import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import random
from collections import defaultdict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class WaferDataset(Dataset):
 
  def __init__(self,df,transform=None,sorted=False,binary=False,just_defects=False):

    #add column for shape, sort by shape if sorted==True.
    df['waferMapShape']=df.waferMap.apply(lambda x: x.shape)
    if sorted==True:
      df.sort_values(by='waferMapShape')

    #wafermap array elements are integers 0,1 or 2. standard so they are floats 0 0.5 and 1 and take np.array; to tensor.
    waferMaps=df['waferMap'].map(lambda x: torch.Tensor([0.5])*torch.Tensor(x).unsqueeze(dim=0))

    failures=df.failureType
    if binary==True:
      failure_dict={'none':0,'Edge-Loc':1,'Loc':1,'Center':1,'Edge-Ring':1,'Scratch':1,'Random':1,'Donut':1,'Near-full':1}
    elif just_defects==True:
      failure_dict={'Edge-Loc':0,'Loc':1,'Center':2,'Edge-Ring':3,'Scratch':4,'Random':5,'Donut':6,'Near-full':7,'none':8}
    else:
      failure_dict={'none':0,'Edge-Loc':1,'Loc':2,'Center':3,'Edge-Ring':4,'Scratch':5,'Random':6,'Donut':7,'Near-full':8}


    self.x=list(waferMaps.values)
    self.y=list(failures.replace(failure_dict).values)

    self.len=len(self.y)
    self.failure_dict=failure_dict
    self.transform=transform

  def __len__(self):
    return self.len
   
  def __getitem__(self,idx):
    x,y=self.x[idx],self.y[idx]
    if self.transform:
      x=self.transform(x)
    return x,y

  def replace_transform(self,transform):
    self.transform=transform


