
import pandas as pd
from huggingface_hub import hf_hub_download


def get_raw_data():
    """
    get raw data from hugging face repo. return as dataframe.
    """
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="LSWMD.pkl")
    return pd.read_pickle(filepath)

def get_processed_data():
    """
    get processed threshold data from hugging face repo. return as train, valid and test dataframe.
    """
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="train.pkl")
    train_df=pd.read_pickle(filepath)
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="valid.pkl")
    valid_df=pd.read_pickle(filepath)
    filepath=hf_hub_download(repo_id="lslattery/wafer-defect-detection", filename="test.pkl")
    test_df=pd.read_pickle(filepath)

    return train_df,valid_df,test_df