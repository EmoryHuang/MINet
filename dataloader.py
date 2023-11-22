from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from dataset import *

import utils


class Poidataloader():
    '''load poi data from Gowalla/Foursquare data file
    '''

    def __init__(self, config):
        self.config = config
        self.database = Path(config.database)
        if not self.database.exists():
            self.database.mkdir()

    def load(self):
        if self.config.dataset.lower() == 'yelp':
            self.checkins = pd.read_feather(self.config.checkins_dataset)
            self.categories = pd.read_feather(self.config.categories_dataset)
            self.friends = pd.read_feather(self.config.friends_dataset)
            if self.config.use_absa:
                absa = torch.load('./Datasets/absa_result.pt',
                                  map_location=torch.device('cpu'))
                self.checkins['absa'] = torch.unbind(absa.cpu())
                self.checkins = self.checkins.drop(columns='text')
        else:
            self.checkins = pd.read_feather(self.config.checkins_dataset)
            self.categories = None
            self.friends = pd.read_feather(self.config.friends_dataset)

        self.static_dataloader()
        return self.checkins

    def create_dataset(self, mode, dataset):
        dataset_train_path = self.database / f'dataset_{dataset}_train.pkl'
        dataset_test_path = self.database / f'dataset_{dataset}_test.pkl'
        dataset_static_path = self.database / f'dataset_{dataset}_static.pkl'
        if dataset_static_path.exists():
            if mode == 'train':
                self.tkg_train, self.kg_train, self.dataset_train = torch.load(
                    dataset_train_path)
            self.tkg_test, self.kg_test, self.dataset_test = torch.load(dataset_test_path)
            setattr(self.config, 'tkg_metadata', self.tkg_test.metadata())
            setattr(self.config, 'kg_metadata', self.kg_test.metadata())
            self.static_dataloader(torch.load(dataset_static_path))
            return

        # Dividing the data set into training and test sets
        checkins_train, checkins_test = utils.split_train_test(self.checkins, 'user', 0.8)
        # checkins_val, checkins_test = utils.split_train_test(checkins_test, 'user, 0.5)

        # Create dataset
        if mode == 'train':
            self.tkg_train = TKGDataset(self.config, checkins_train)
            self.kg_train = StaticKGDataset(self.config).generate_KG(
                checkins_train, self.friends, self.categories)
            self.dataset_train = PoiDataset(self.config, checkins_train)
            torch.save([self.tkg_train, self.kg_train, self.dataset_train],
                       dataset_train_path)

        self.tkg_test = TKGDataset(self.config, checkins_test)
        self.kg_test = StaticKGDataset(self.config).generate_KG(
            checkins_test, self.friends, self.categories)
        self.dataset_test = PoiDataset(self.config, checkins_test)
        torch.save([self.tkg_test, self.kg_test, self.dataset_test], dataset_test_path)

        self.dataset_static = torch.LongTensor([
            self.config.max_user_num, self.config.max_loc_num, self.config.max_geo_num,
            self.config.max_cate_num
        ])
        torch.save(self.dataset_static, dataset_static_path)
        setattr(self.config, 'tkg_metadata', self.tkg_test.metadata())
        setattr(self.config, 'kg_metadata', self.kg_test.metadata())

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=True,
        )

    def train_kgloader(self):
        return DataLoader(
            dataset=self.tkg_train,
            batch_size=self.config.tkg_batch_size,
            pin_memory=True,
            shuffle=False,
        ), self.kg_train

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def val_kgloader(self):
        return DataLoader(
            dataset=self.tkg_test,
            batch_size=self.config.tkg_batch_size,
            shuffle=False,
        ), self.kg_test

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def test_kgloader(self):
        return DataLoader(
            dataset=self.tkg_test,
            batch_size=self.config.tkg_batch_size,
            shuffle=False,
        ), self.kg_test

    def user_count(self):
        return self.checkins['user'].unique().size

    def location_count(self):
        return self.checkins['location'].unique().size

    def geohash_count(self):
        return self.checkins['geohash'].unique().size

    def category_count(self):
        if self.config.use_cate:
            return self.categories['category'].unique().size
        else:
            return 1

    def checkins_count(self):
        return len(self.checkins)

    def static_dataloader(self, dataset_static=None):
        if dataset_static is None:
            dataset_static = [
                self.user_count() + 1,
                self.location_count() + 1,
                self.geohash_count() + 1,
                self.category_count() + 1
            ]
        setattr(self.config, 'max_user_num', int(dataset_static[0]))
        setattr(self.config, 'max_loc_num', int(dataset_static[1]))
        setattr(self.config, 'max_geo_num', int(dataset_static[2]))
        setattr(self.config, 'max_cate_num', int(dataset_static[3]))
        setattr(self.config, 'max_time_num', 24 + 1)
        setattr(self.config, 'max_week_num', 7 + 1)
