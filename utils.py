import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import pygeohash as gh


def init_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create consol handler
    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.INFO)

    # set format
    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s line:%(lineno)d process:%(process)d] %(levelname)s: %(message)s'
    )
    consol_handler.setFormatter(formatter)
    logger.addHandler(consol_handler)
    return logger


def geohash_encode(checkins, precision=5):
    '''Encode the latitude and longitude use geohash
    '''
    checkins['geohash'] = checkins.apply(
        lambda x: gh.encode(x['latitude'], x['longitude'], precision=precision), axis=1)
    return checkins


def cal_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    R = 6371  # Radius of the earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    distance = 2 * np.arcsin(np.sqrt(a)) * R
    return distance


def split_train_test(df, column, train_rate):

    def _split_train_test(df):
        n_train = int(len(df) * train_rate)
        df_train.append(df.iloc[:n_train])
        df_test.append(df.iloc[n_train - 1:])

    df_train, df_test = [], []
    _ = df.groupby(column).apply(_split_train_test)
    df_train = pd.concat(df_train).reset_index(drop=True)
    df_test = pd.concat(df_test).reset_index(drop=True)
    return df_train, df_test


def calculate_acc(pred, labels):
    '''calculate acc
    
    `result[0, 1, 2, 3]` represent `recall@1`, `recall@5`, `recall@10`, `MAP` respectively
    '''
    # pred shape: (batch_size, max_sequence_length, max_loc_num)
    # labels shape: (batch_size, max_sequence_length)

    # pred shape: (batch_size * max_sequence_length, max_loc_num)
    pred = pred.view(-1, pred.shape[2])
    # labels shape: (1, batch_size * max_sequence_length)
    labels = labels.view(-1).unsqueeze(0)

    result = torch.zeros(5, labels.shape[1], device=pred.device)
    # calculate recall@k (k=1, 5, 10, 20)
    # get topk predict pois
    pred_val, pred_poi = pred.topk(20, dim=1, sorted=True)
    recall = torch.stack([labels == pred_poi[:, i] for i in range(20)])
    result[0] = recall[:1].sum(dim=0)
    result[1] = recall[:5].sum(dim=0)
    result[2] = recall[:10].sum(dim=0)
    result[3] = recall[:20].sum(dim=0)

    # calculate MRR
    # find the score of the label corresponding to the POI
    score = pred.gather(dim=1, index=labels.T)
    result[4] = 1 / (1 + (pred > score).sum(dim=1))

    return result