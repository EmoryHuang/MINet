from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from utils import *


class Trainer:
    '''
    
    Args:
    -------
    config: Config
        configuration file
    logger: logging = None
        logging object
    gpu: int = -1
        Specify the GPU device. if `gpu=-1` then use CPU.

    '''

    def __init__(self, config, logger=None, gpu=-1):
        self.config = config
        self.logger = logger if logger is not None else init_logger()

        if gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(self.config.model_dir)
        if not self.model_dir.exists():
            self.model_dir.mkdir()

    def train(self, model, dataloader):
        # prepare optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # prepare dataloader
        train_dl = dataloader.train_dataloader()
        train_tkg_dl, train_static_kg = dataloader.train_kgloader()
        val_dl = dataloader.val_dataloader()
        val_tkg_dl, val_static_kg = dataloader.val_kgloader()
        # prepare model
        model = model.to(self.device)

        self.logger.info('start training...')
        self.logger.info(f'device: {self.device}')
        for epoch in range(self.config.epochs):
            # train loop
            self._train_epoch(epoch, model, train_tkg_dl, train_static_kg, train_dl,
                              optimizer, scheduler, criterion)

            # valid loop
            self._val_epoch(epoch, model, val_tkg_dl, val_static_kg, val_dl, criterion)

            model_path = self.model_dir / f"model_{epoch+1}.pkl"
            torch.save(model.state_dict(), model_path)
        self.logger.info('training done!')

    def _train_epoch(self, epoch, model, train_tkg_dl, train_static_kg, train_dl,
                     optimizer, scheduler, criterion):
        model.train()

        train_loss = []
        pbar = tqdm(train_dl, total=len(train_dl))
        for idx, dl in enumerate(pbar):
            if self.config.use_absa:
                user, traj, _, time, week, absa, tkg_idx, loc_user_group, geo_user_group, label_traj, _ = dl
                absa = absa.to(self.device)
            else:
                user, traj, _, time, week, tkg_idx, loc_user_group, geo_user_group, label_traj, _ = dl
                absa = None

            user = user.to(self.device)
            traj = traj.to(self.device)
            time = time.to(self.device)
            week = week.to(self.device)
            tkg_idx = tkg_idx.to(self.device)
            loc_user_group = loc_user_group.to(self.device)
            geo_user_group = geo_user_group.to(self.device)
            label_traj = label_traj.to(self.device)
            train_static_kg = train_static_kg.to(self.device)

            optimizer.zero_grad()
            pred_poi, loss_static = model(user, traj, time, week, absa, train_tkg_dl,
                                          train_static_kg, tkg_idx, loc_user_group,
                                          geo_user_group)

            # calculate the loss
            loss_poi = criterion(pred_poi.permute(0, 2, 1), label_traj)
            loss_all = self.config.lamb * loss_static + (1 - self.config.lamb) * loss_poi
            # loss_all = loss_poi

            loss_all.backward()
            optimizer.step()
            train_loss.append(loss_all.item())

            # update pbar
            pbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            pbar.set_postfix(loss=f'{np.mean(train_loss):.4f}',
                             lr=scheduler.get_last_lr()[0])
        scheduler.step()

    @torch.no_grad()
    def _val_epoch(self, epoch, model, val_tkg_dl, val_static_kg, val_dl, criterion):
        if (epoch + 1) % 1 != 0:
            return

        model.eval()
        val_loss, val_acc = [], []
        vbar = tqdm(val_dl, desc='valid', total=len(val_dl))
        for idx, dl in enumerate(vbar):
            if self.config.use_absa:
                user, traj, _, time, week, absa, tkg_idx, loc_user_group, geo_user_group, label_traj, _ = dl
                absa = absa.to(self.device)
            else:
                user, traj, _, time, week, tkg_idx, loc_user_group, geo_user_group, label_traj, _ = dl
                absa = None

            user = user.to(self.device)
            traj = traj.to(self.device)
            time = time.to(self.device)
            week = week.to(self.device)
            tkg_idx = tkg_idx.to(self.device)
            loc_user_group = loc_user_group.to(self.device)
            geo_user_group = geo_user_group.to(self.device)
            label_traj = label_traj.to(self.device)
            val_static_kg = val_static_kg.to(self.device)

            pred_poi, loss_static = model(user, traj, time, week, absa, val_tkg_dl,
                                          val_static_kg, tkg_idx, loc_user_group,
                                          geo_user_group)

            # calculate the loss
            loss_poi = criterion(pred_poi.permute(0, 2, 1), label_traj)
            loss_all = self.config.lamb * loss_static + (1 - self.config.lamb) * loss_poi
            # loss_all = loss_poi

            val_acc.append(calculate_acc(pred_poi, label_traj))
            val_loss.append(loss_all.item())

            # update pbar
            mean_acc = torch.concat(val_acc, dim=1).mean(dim=1).cpu().tolist()
            mean_acc = [round(acc, 4) for acc in mean_acc]
            vbar.set_postfix(val_loss=f'{np.mean(val_loss):.4f}', acc=mean_acc)

    @torch.no_grad()
    def test(self, model, dataloader, model_path):
        # prepare dataloader
        test_dl = dataloader.test_dataloader()
        test_tkg_dl, test_static_kg = dataloader.test_kgloader()

        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_acc = []
        tbar = tqdm(test_dl, desc='test', total=len(test_dl))
        self.logger.info('start testing...')
        for idx, dl in enumerate(tbar):
            if self.config.use_absa:
                user, traj, _, time, week, absa, tkg_idx, loc_user_group, geo_user_group, label_traj, _ = dl
                absa = absa.to(self.device)
            else:
                user, traj, _, time, week, tkg_idx, loc_user_group, geo_user_group, label_traj, _ = dl
                absa = None

            user = user.to(self.device)
            traj = traj.to(self.device)
            time = time.to(self.device)
            week = week.to(self.device)
            tkg_idx = tkg_idx.to(self.device)
            loc_user_group = loc_user_group.to(self.device)
            geo_user_group = geo_user_group.to(self.device)
            label_traj = label_traj.to(self.device)
            test_static_kg = test_static_kg.to(self.device)

            pred_poi, loss_static = model(user, traj, time, week, absa, test_tkg_dl,
                                          test_static_kg, tkg_idx, loc_user_group,
                                          geo_user_group)

            test_acc.append(calculate_acc(pred_poi, label_traj))

            # update pbar
            mean_acc = torch.concat(test_acc, dim=1).mean(dim=1).cpu().tolist()
            mean_acc = [round(acc, 4) for acc in mean_acc]
            tbar.set_postfix(acc=mean_acc)

        self.logger.info('testing done.')
        self.logger.info('-------------------------------------')
        self.logger.info('test result:')
        self.logger.info(f'Acc@1: {mean_acc[0]}')
        self.logger.info(f'Acc@5: {mean_acc[1]}')
        self.logger.info(f'Acc@10: {mean_acc[2]}')
        self.logger.info(f'Acc@20: {mean_acc[3]}')
        self.logger.info(f'MRR: {mean_acc[4]}')