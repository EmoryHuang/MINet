from pathlib import Path

import numpy as np
import torch
import utils
import pandas as pd
from args import get_pretraining_args
from dataset import TextDataset
from model import TEXTClassification
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()

# load config file
args = get_pretraining_args()

# init logger
logger = utils.init_logger()

# init seed
utils.init_seed(3407)


class Pretraining:

    def __init__(self, config, logger=None) -> None:
        self.config = config
        self.logger = logger if logger is not None else utils.init_logger()

        if self.config.gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(self.config.model_dir)
        if not self.model_dir.exists():
            self.model_dir.mkdir()

    def train(self, model):
        # prepare optimizer and criterion
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=self.config.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        early_stopping = utils.EarlyStopping(self.model_dir)

        # prepare dataloader
        self.logger.info('start creating dataset...')
        dl_train, dl_val = Pretraining.create_dataset(self.config)
        self.logger.info('creating dataset done!')

        # prepare model
        self.logger.info('start loading model...')
        model = model.to(self.device)
        self.logger.info('loading model done!')

        self.logger.info('start training...')
        for epoch in range(self.config.epochs):
            # train loop
            self._train_epoch(epoch, model, dl_train, optimizer, scheduler)

            # valid loop
            self._val_epoch(model, dl_val)

            early_stopping(self.val_loss, model)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        self.logger.info('training done!')

    def _train_epoch(self, epoch, model, dl_train, optimizer, scheduler):
        model.train()
        train_loss = []
        pbar = tqdm(dl_train, total=len(dl_train))
        for idx, dl in enumerate(pbar):
            text, aspect, polarity = dl
            aspect = aspect.to(self.device)
            polarity = polarity.to(self.device)

            optimizer.zero_grad()
            loss, aspect_res, polarity_res = model(text, aspect, polarity)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # update pbar
            pbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            pbar.set_postfix(loss=np.mean(train_loss), lr=scheduler.get_last_lr()[0])

        scheduler.step()

    @torch.no_grad()
    def _val_epoch(self, model, dl_val):
        model.eval()
        val_loss = []
        aspect_acc, polarity_acc = [], []

        vbar = tqdm(dl_val, desc='valid', total=len(dl_val))
        for idx, dl in enumerate(vbar):
            text, aspect, polarity = dl
            aspect = aspect.to(self.device)
            polarity = polarity.to(self.device)

            loss, aspect_res, polarity_res = model(text, aspect, polarity)
            val_loss.append(loss.item())

            # caluate acc
            aspect_acc.append(Pretraining.cal_acc(aspect_res, aspect))
            polarity_acc.append(Pretraining.cal_acc(polarity_res, polarity))
            mean_aspect_acc = torch.concat(aspect_acc).float().mean().cpu()
            mean_polarity_acc = torch.concat(polarity_acc).float().mean().cpu()
            mean_aspect_acc = round(mean_aspect_acc.item(), 4)
            mean_polarity_acc = round(mean_polarity_acc.item(), 4)

            # update vbar
            self.val_loss = np.mean(val_loss)
            vbar.set_postfix(val_loss=np.mean(val_loss),
                             aspect_acc=mean_aspect_acc,
                             polarity_acc=mean_polarity_acc)

    @torch.no_grad()
    def test(self, model, text):
        dataset = TextDataset(text)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)

        model = model.to(self.device)
        model.load_state_dict(torch.load('./Model/absa.pt', map_location=self.device))
        model.eval()

        result = []
        self.logger.info('start testing...')
        tbar = tqdm(dataloader, desc='testing', total=len(dataloader))
        for test_data in tbar:
            aspect_res, logist = model(test_data, device=self.device)
            logist = torch.sigmoid(logist)
            result.append(logist.view(-1, 6))
        # result shape: (batch_size, 3, 2)
        self.logger.info('testing done.')
        return torch.concat(result)

    @staticmethod
    def cal_acc(pred, label):
        return torch.argmax(pred, dim=1) == label

    @staticmethod
    def create_dataset(config):
        raw_data = open(config.dataset, 'r').readlines()
        data = []
        for line in raw_data:
            text, aspect, polarity = line.strip().split('\t')
            data.append([text, int(aspect), int(polarity)])

        td_data = TextDataset(data)
        n_train = int(len(td_data) * 0.8)
        n_test = len(td_data) - n_train
        ds_train, ds_test = random_split(td_data, [n_train, n_test])

        dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
        dl_test = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)
        return dl_train, dl_test


def main():
    pretraining = Pretraining(args, logger)
    logger.info(f'Pretrain model: {args.pretrain_model}')
    model = TEXTClassification.from_pretrained(args.pretrain_model, num_labels=3)
    if args.mode == 'train':
        pretraining.train(model)
    else:
        # result shape: (batch_size, 3, 2)
        # result[i][0]: food score
        # result[i][1]: price score
        # result[i][2]: service score
        # result[i][j][0]: negative score
        # result[i][j][1]: positive score
        checkins = pd.read_feather('./Datasets/checkins_30.feather')
        checkins['text'] = checkins['text'].apply(lambda x: x[:512])
        text = checkins['text'].tolist()
        result = pretraining.test(model, text)
        torch.save(result, './Datasets/absa_result.pt')


if __name__ == '__main__':
    main()

# train
# python pretraining.py --gpu=0
# nohup python pretraining.py --gpu=0 > ./absa.log 2>&1 &

# test
# nohup python pretraining.py --mode=test --gpu=2 > ./absa.log 2>&1 &