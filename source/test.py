# -*- coding: utf-8 -*-

import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from utils import create_logger, seed_set
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, save_best_checkpoint, load_best_result
from dataset import build_loader
from loss import bulid_loss
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="codes for HiGNN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../configs/bbbp.yaml",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for training")
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    args = parser.parse_args()
    cfg = get_config(args)

    return args, cfg


@torch.no_grad()
def validate(cfg, model, criterion, dataloader, epoch, device, logger, eval_mode=False):
    model.eval()

    losses = []
    y_pred_list = {}
    y_label_list = {}

    for data in dataloader:
        data = data.to(device)
        output = model(data)
        if isinstance(output, tuple):
            output, vec1, vec2 = output
        else:
            output, vec1, vec2 = output, None, None
        loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                loss += criterion[i](y_pred, y_label, vec1, vec2)
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            else:
                y_pred = output[:, i]
                y_label = data.y[:, i]
                loss += criterion(y_pred, y_label, vec1, vec2)
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            losses.append(loss.item())

    # Compute metric
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while validating')

            if nan:
                val_results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array(losses).mean()
    if eval_mode:
        logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> '
                    f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
        return val_results

    return val_loss, avg_val_results


def test(cfg, logger):
    seed_set(cfg.SEED)
    # step 1: dataloder loading, get number of tokens
    train_loader, val_loader, test_loader, weights = build_loader(cfg, logger)
    # step 2: model loading
    model = build_model(cfg)
    logger.info(model)
    # device mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # step 3: optimizer loading
    optimizer = build_optimizer(cfg, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # step 4: lr_scheduler loading
    lr_scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    if weights is not None:
        criterion = [bulid_loss(cfg, torch.Tensor(w).to(device)) for w in weights]
    else:
        criterion = bulid_loss(cfg)
    best_epoch, best_score = 0, 0 if cfg.DATA.TASK_TYPE == 'classification' else float('inf')
    if cfg.TRAIN.RESUME:
        best_epoch, best_score = load_checkpoint(cfg, model, optimizer, lr_scheduler, logger)
        score = validate(cfg, model, criterion, test_loader, best_epoch, device, logger=logger,eval_mode=True)
    logger.info(f'Seed {cfg.SEED} ==> test {cfg.DATA.METRIC} = {np.nanmean(score):.3f}')
    return score


if __name__ == "__main__":
    _, cfg = parse_args()

    logger = create_logger(cfg)

    # print config
    logger.info(cfg.dump())
    # print device mode
    if torch.cuda.is_available():
        logger.info('GPU mode...')
    else:
        logger.info('CPU mode...')
    # training
    test(cfg, logger)



