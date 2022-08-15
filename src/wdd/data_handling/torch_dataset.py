import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class WaferDataset(Dataset):
 
  def __init__(self,filepath):

    df=pd.read_pickle(filepath)
 
    waferMaps=df['waferMap'].map(lambda x: torch.Tensor(x).unsqueeze(dim=0))
    failures=df.failureType
    failure_dict={'none':0,'Edge-Loc':1,'Loc':2,'Center':3,'Edge-Ring':4,'Scratch':5,'Random':6,'Donut':7,'Near-full':8}

    self.x=list(waferMaps.values)
    self.y=list(failures.replace(failure_dict).values)

    self.len=len(self.y)
    self.file_name=filepath
    self.failure_dict=failure_dict
  def __len__(self):
    return self.len
   
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]