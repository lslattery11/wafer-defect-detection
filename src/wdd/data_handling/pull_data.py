
import pandas as pd
from huggingface_hub import hf_hub_download
from typing import Tuple


def get_raw_data():
    """
    get raw data from hugging face repo. return as dataframe.
    """
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="LSWMD.pkl")
    return filepath

def get_processed_data() -> Tuple[str]:
    """
    get processed threshold data from hugging face repo. return as train and test dataframe.
    """
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="train.pkl",repo_type='dataset')
    train_df=pd.read_pickle(filepath)
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="test.pkl",repo_type='dataset')
    test_df=pd.read_pickle(filepath)

    return (train_df,test_df)