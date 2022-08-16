import pandas as pd
from pandas import DataFrame
import numpy as np

def threshold_data(
    df : DataFrame,
    size_threshold : int ,
    ) -> DataFrame:
    """
    compute wafer map size, keep data above size threshold and rescale the data.
    """
    #compute size, apply threshold cut.
    size_series=df.apply(lambda x : x.waferMap.size, axis=1)
    df['waferMapSize']=size_series

    threshold_df=df[(df['waferMapSize'] > size_threshold)]
    #clean up data by squeezing labels from array -> strings and removing 
    temp_failure=threshold_df.apply(lambda x: np.squeeze(x.failureType)[()],axis=1)
    threshold_df['failureType']=temp_failure
    threshold_df=threshold_df[(temp_failure.apply(type).eq(np.str_))]
    #keep only the data,data size and labels.
    threshold_df=threshold_df[['waferMap','waferMapSize','failureType']]
    return threshold_df

