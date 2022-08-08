import os
import subprocess
import sys
from huggingface_hub import hf_hub_download
import pandas as pd


def get_raw_defect_data():
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="LSWMD.pkl")
    return pd.read_pickle(filepath)

def get_processed_data():
    
    return