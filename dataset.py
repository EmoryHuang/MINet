import numpy as np
import torch
from generate_triple import *
from torch import LongTensor as LT
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from tqdm import tqdm


class TextDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TKGDataset(Dataset):

    def __init__(self, config, checkins):
        super(TKGDataset, self).__init__()
        self.config = config

        self.temporal_split(checkins)

    def temporal_split(self, checkins):
        # group by year-month
        if self.config.group == '10D':
            grouped = checkins.groupby(pd.Grouper(key='time', freq='10D'))
        elif self.config.group == '15D':
            grouped = checkins.groupby(pd.Grouper(key='time', freq='15D'))
        elif self.config.group == 'month':
            grouped = checkins.groupby(
                [checkins['time'].dt.year, checkins['time'].dt.month])
        elif self.config.group == 'quarter':
            grouped = checkins.groupby(
                [checkins['time'].dt.year, checkins['time'].dt.quarter])
        df_kgs = [val for key, val in grouped]
        tbar = tqdm(df_kgs, desc='Generate TKG')
        self.tkg = [self.generate_KG(df) for df in tbar]

    def generate_KG(self, checkins):
        visited_triple = generate_visited_triple(checkins)
        followby_triple = generate_followby_triple(checkins)

        #############################################################
        # generate knowledge graph
        #############################################################
        data = HeteroData()
        # generate node
        data['user'].x = torch.arange(self.config.max_user_num)
        data['location'].x = torch.arange(self.config.max_loc_num)

        # generate edge
        data['user', 'visited', 'location'].edge_index = \
            LT(visited_triple[['subject', 'object']].values).T
        data['location', 'followby', 'location'].edge_index = \
            LT(followby_triple[['subject', 'object']].values).T
        return data

    def metadata(self):
        return self.tkg[0].metadata()

    def __len__(self):
        return len(self.tkg)

    def __getitem__(self, idx):
        return self.tkg[idx]


class StaticKGDataset:

    def __init__(self, config) -> None:
        self.config = config

    def generate_KG(self, checkins, friends, categories):
        #############################################################
        # generate triple
        #############################################################
        nearby_triple = generate_nearby_triple(checkins)
        locateat_triple = generate_locateat_triple(checkins)
        if self.config.use_social:
            social_triple = generate_social_triple(friends)
        if self.config.use_cate:
            cateof_triple = generate_cateof_triple(categories)

        #############################################################
        # generate knowledge graph
        #############################################################
        data = HeteroData()
        # generate node
        data['user'].x = torch.arange(self.config.max_user_num)
        data['location'].x = torch.arange(self.config.max_loc_num)
        data['area'].x = torch.arange(self.config.max_geo_num)
        if self.config.use_cate:
            data['category'].x = torch.arange(self.config.max_cate_num)

        # generate edge
        data['location', 'nearby', 'location'].edge_index = \
            LT(nearby_triple[['subject', 'object']].values).T
        data['location', 'locateat', 'area'].edge_index = \
            LT(locateat_triple[['subject', 'object']].values).T
        if self.config.use_social:
            data['user', 'social', 'user'].edge_index = \
                LT(social_triple[['subject', 'object']].values).T
        if self.config.use_cate:
            data['location', 'cateof', 'category'].edge_index = \
                LT(cateof_triple[['subject', 'object']].values).T
        return data


class PoiDataset(Dataset):

    def __init__(self, config, checkins):
        super(PoiDataset, self).__init__()
        self.config = config
        self.checkins = checkins.copy()

        self.checkins_user_split()
        self.checkins_location_split()

    def checkins_user_split(self):
        '''split the checkins (pd.DataFrame) and collect.
        '''
        checkins = self.checkins
        # checkins['timestamp'] = checkins['time'].astype('int') // 10**9

        # generate tkg index mapping
        # {(year, month) : index}
        grouped = checkins.groupby([checkins['time'].dt.year, checkins['time'].dt.month])
        mapping = dict(zip(grouped.groups.keys(), range(len(grouped))))

        def collect_fn(df):
            # Groupby user_id and collect trajs, times etc.
            max_sequence_length = self.config.max_sequence_length
            cnt, remain = divmod(len(df) - 1, max_sequence_length)
            df = df[:cnt * max_sequence_length + 1]

            user.append(LT(df['user'][:-1].values))
            traj.append(LT(df['location'][:-1].values))
            geo.append(LT(df['geohash'][:-1].values))
            time.append(LT(df['time'][:-1].dt.hour.values))
            week.append(LT(df['time'][:-1].dt.dayofweek.values))
            if self.config.use_absa:
                absa_list = df['absa'][:-1].tolist()
                if len(absa_list):
                    absa.append(torch.stack(absa_list))

            date = list(zip(df['time'][:-1].dt.year, df['time'][:-1].dt.month))
            tkg_idx.append(LT(list(map(lambda x: mapping[x], date))))

            label_traj.append(LT(df['location'][1:].values))
            label_geo.append(LT(df['geohash'][1:].values))

        max_sequence_length = self.config.max_sequence_length
        user, traj, geo, time, week, absa = [], [], [], [], [], []
        tkg_idx = []
        label_traj, label_geo = [], []
        _ = checkins.groupby('user').apply(collect_fn)

        self.user = torch.concat(user).view(-1, max_sequence_length)
        self.traj = torch.concat(traj).view(-1, max_sequence_length)
        self.geo = torch.concat(geo).view(-1, max_sequence_length)
        if self.config.use_absa:
            self.absa = torch.concat(absa).view(-1, max_sequence_length, 6)
        self.tkg_idx = torch.concat(tkg_idx).view(-1, max_sequence_length)
        self.label_traj = torch.concat(label_traj).view(-1, max_sequence_length)
        self.label_geo = torch.concat(label_geo).view(-1, max_sequence_length)

        self.time = torch.concat(time).view(-1, max_sequence_length)
        self.week = torch.concat(week).view(-1, max_sequence_length)

    def checkins_location_split(self):

        def collect_fn(df, type):
            if type == 'location':
                loc_user_group[df.iloc[0][type]] = LT(df['user'].unique())
            else:
                geo_user_group[df.iloc[0][type]] = LT(df['user'].unique())

        loc_user_group = [LT(0)] * self.config.max_loc_num
        geo_user_group = [LT(0)] * self.config.max_geo_num
        _ = self.checkins.groupby('location').apply(collect_fn, type='location')
        _ = self.checkins.groupby('geohash').apply(collect_fn, type='geohash')
        self.loc_user_group = pad_sequence(loc_user_group, batch_first=True)
        self.geo_user_group = pad_sequence(geo_user_group, batch_first=True)

    def __len__(self):
        return self.user.size(0)

    def __getitem__(self, idx):
        user = self.user[idx]
        traj = self.traj[idx]
        geo = self.geo[idx]
        time = self.time[idx]
        week = self.week[idx]

        tkg_idx = self.tkg_idx[idx]
        loc_user_group = self.loc_user_group[traj]
        geo_user_group = self.geo_user_group[geo]

        label_traj = self.label_traj[idx]
        label_geo = self.label_geo[idx]
        if self.config.use_absa:
            absa = self.absa[idx]
            return user, traj, geo, time, week, absa, tkg_idx, loc_user_group, geo_user_group, label_traj, label_geo
        else:
            return user, traj, geo, time, week, tkg_idx, loc_user_group, geo_user_group, label_traj, label_geo