import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def exclude_low_variance(feature_value, threshold=0.01):
    """

    :param feature_value:
    :param threshold:
    :return:
    """
    # case_id = feature_frame['case_ids']
    # label = feature_frame['label']

    var = feature_value.var()
    df_variance = var.to_frame(name='variance').transpose()
    new_feature = feature_value.loc[:, df_variance.iloc[0] > threshold]

    return new_feature


def min_max(feature_value):
    """

    :param feature_frame:
    :return:
    """
    scaler = MinMaxScaler()
    new_feature = scaler.fit_transform(feature_value)
    new_feature = pd.DataFrame(new_feature, columns=feature_value.columns)
    return new_feature


def run_preprocessing(feature_frame, drop_columns, variance_threshold=0.01, method=0):
    """

    :param feature_frame:
    :param drop_columns: columns names to drop to leave the feature only such as ['case_ids','labels']
    :param variance_threshold:
    :param method:
    :return:
    """
    feature_value = feature_frame.drop(drop_columns, axis=1)
    feature_after_LR = exclude_low_variance(feature_value, variance_threshold)
    if method == 0:
        new_feature = (feature_after_LR - feature_after_LR.mean(axis=0)) / feature_after_LR.std(axis=0)
    else:
        new_feature = min_max(feature_after_LR)
    new_feature[drop_columns] = feature_frame[drop_columns]
    return new_feature
