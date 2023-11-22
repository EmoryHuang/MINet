# node type
# user || location || area || category
#
# relation type
# 0 represent `Visited` relation
# 1 represent `FollowBy` relation
# 2 represent `NearBy` relation
# 3 represent `LocateAt` relation
# 4 represent `Social` relation
# 5 represent `CateOf` relation

import numpy as np
import pandas as pd
import utils
from tqdm import tqdm


def generate_visited_triple(checkins):
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    df[['subject', 'object']] = checkins[['user', 'location']]
    df['predicate'] = 0
    df = df.dropna().drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df


def generate_followby_triple(checkins):
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    checkins = checkins.sort_values(['user', 'time'])
    checkins.reset_index(drop=True, inplace=True)
    df['subject'] = checkins['location'][:-1].values
    df['object'] = checkins['location'][1:].values
    df['predicate'] = 1
    # drop last visiting
    drop_index = checkins['user'].drop_duplicates(keep='last').index[:-1]
    df = df.drop(index=drop_index)
    df = df.dropna().drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df


def generate_nearby_triple(checkins, thr=3, k=50):
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    poi = checkins[['location', 'latitude', 'longitude']].drop_duplicates()
    # calualate distance and filter by threshold and topK
    subject = np.array([], dtype=int)
    object = np.array([], dtype=int)
    for i in tqdm(range(len(poi)), desc='Generate triple'):
        cur_loc, cur_lat, cur_lon = poi.iloc[i]
        dis = utils.cal_distance(cur_lat, cur_lon, poi['latitude'], poi['longitude'])
        accept_index = dis[dis < thr].nsmallest(k + 1)[1:].index
        accept_poi = poi[poi.index.isin(accept_index)]['location'].values
        subject = np.concatenate([subject, np.repeat(int(cur_loc), accept_poi.size)])
        object = np.concatenate([object, accept_poi])
    df['subject'] = subject
    df['object'] = object
    df['predicate'] = 2
    df = df.dropna().drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df


def generate_locateat_triple(checkins):
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    df[['subject', 'object']] = checkins[['location', 'geohash']]
    df['predicate'] = 3
    df = df.dropna().drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df


def generate_social_triple(friends):
    friends = friends.copy()
    friends.columns = ['subject', 'object']
    friends['predicate'] = 4
    friends = friends[['subject', 'predicate', 'object']]
    return friends


def generate_cateof_triple(categories):
    categories = categories.copy()
    categories.columns = ['subject', 'object']
    categories['predicate'] = 5
    categories = categories[['subject', 'predicate', 'object']]
    return categories


def generate_triple(checkins, friends, categories):
    max_user_num = checkins['user'].unique().size + 1
    max_loc_num = checkins['location'].unique().size + 1
    max_geo_num = checkins['geohash'].unique().size + 1
    max_cate_num = categories['category'].unique().size + 1

    # generate triple
    visited_triple = generate_visited_triple(checkins)
    followby_triple = generate_followby_triple(checkins)
    nearby_triple = generate_nearby_triple(checkins)
    locateat_triple = generate_locateat_triple(checkins)
    social_triple = generate_social_triple(friends)
    cateof_triple = generate_cateof_triple(categories)

    # fix id
    # user || location || area || category
    visited_triple['object'] += max_user_num
    followby_triple[['subject', 'object']] += max_user_num
    nearby_triple[['subject', 'object']] += max_user_num
    locateat_triple[['subject', 'object']] += max_user_num
    locateat_triple['object'] += max_loc_num
    cateof_triple[['subject', 'object']] += max_user_num
    cateof_triple['object'] += max_loc_num + max_geo_num

    # split train/val/test
    visited_triple_train, visited_triple_test = utils.split_train_test(
        visited_triple, 'subject', 0.8)
    visited_triple_val, visited_triple_test = utils.split_train_test(
        visited_triple_test, 'subject', 0.5)

    followby_triple_train, followby_triple_test = utils.split_train_test(
        followby_triple, 'subject', 0.8)
    followby_triple_val, followby_triple_test = utils.split_train_test(
        followby_triple_test, 'subject', 0.5)

    nearby_triple_train, nearby_triple_test = utils.split_train_test(
        nearby_triple, 'subject', 0.8)
    nearby_triple_val, nearby_triple_test = utils.split_train_test(
        nearby_triple_test, 'subject', 0.5)

    locateat_triple_train, locateat_triple_test = utils.split_train_test(
        locateat_triple, 'subject', 0.8)
    locateat_triple_val, locateat_triple_test = utils.split_train_test(
        locateat_triple_test, 'subject', 0.5)

    social_triple_train, social_triple_test = utils.split_train_test(
        social_triple, 'subject', 0.8)
    social_triple_val, social_triple_test = utils.split_train_test(
        social_triple_test, 'subject', 0.5)

    cateof_triple_train, cateof_triple_test = utils.split_train_test(
        cateof_triple, 'subject', 0.8)
    cateof_triple_val, cateof_triple_test = utils.split_train_test(
        cateof_triple_test, 'subject', 0.5)

    # concat
    triple_train = pd.concat([
        visited_triple_train,
        followby_triple_train,
        nearby_triple_train,
        locateat_triple_train,
        social_triple_train,
        cateof_triple_train,
    ])
    triple_val = pd.concat([
        visited_triple_val,
        followby_triple_val,
        nearby_triple_val,
        locateat_triple_val,
        social_triple_val,
        cateof_triple_val,
    ])
    triple_test = pd.concat([
        visited_triple_test,
        followby_triple_test,
        nearby_triple_test,
        locateat_triple_test,
        social_triple_test,
        cateof_triple_test,
    ])
    triple_train.reset_index(drop=True, inplace=True)
    triple_val.reset_index(drop=True, inplace=True)
    triple_test.reset_index(drop=True, inplace=True)
    return triple_train, triple_val, triple_test


if __name__ == '__main__':
    pass