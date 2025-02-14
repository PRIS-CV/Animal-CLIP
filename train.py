import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, all_gather, get_map, get_animal, compute_F1, lt_map, get_relation
from datasets.tools import pack_pathway_output
from utils.visualize import visualize
from utils.loss import ResampleLoss, AsymmetricLoss
from datasets.build import build_dataloader, build_video_prompt_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending, one_hot
from utils.config import get_config
from models import model_builder
from models.prompt import text_prompt
import math
import clip
import requests
import coop
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score

from torch.utils.tensorboard import SummaryWriter

from torch.utils import checkpoint

import pandas as pd

writer = SummaryWriter()

def get_para_num(model):
    lst = []
    for para in model.parameters():
        lst.append(para.nelement())
    print(f"total paras number: {sum(lst)}")
    
    
def get_trainable_para_num(model):
    lst = []
    for name, para in model.named_parameters():
    # for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, animal_labels, config, mixup_fn, train_data, logger):
    get_para_num(model)
    get_trainable_para_num(model)
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    texts = text_labels.cuda(non_blocking=True)
    # animals = animal_labels.cuda(non_blocking=True)
    edges = get_relation(config.DATA.LABEL_LIST, config.DATA.RELATION_FILE)
    edges = torch.tensor(edges).cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):
        
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        animal_pred = batch_data["animal"].cuda(non_blocking=True)
        mid_frame = batch_data["mid_frame"].cuda(non_blocking=True)
        
        # label_id = label_id.reshape(-1)
        images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])
        
        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)
            
        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        
        if config.MODEL.MODEL_NAME == 'XCLIP':
            output, _ = model(images, texts, animal_labels, animal_pred, edges)
            
        elif config.MODEL.MODEL_NAME == 'VideoPrompt':
        # for id in action_id:
            name_map = pd.read_csv(config.DATA.LABEL_LIST).values.tolist()
            inp_actionlist = [name_map[i][1] for i in range(len(name_map))]
            output = model(images, inp_actionlist)
            
        else:
            images = pack_pathway_output(config, images)
            output = model(images)
        
        total_loss = criterion(output, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0': # 纯FP32训练
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.requires_grad_(True)
                scaled_loss.backward()
        else:
            total_loss.backward()
            
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                if config.TRAIN.LR_POLICY == 'cosineannealingwarmrestarts':
                    lr_scheduler.step(epoch * num_steps + idx)
                else:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            if config.TRAIN.LR_POLICY == 'cosineannealingwarmrestarts':
                    lr_scheduler.step(epoch * num_steps + idx)
            else:
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    writer.add_scalar('loss', tot_loss_meter.avg, epoch)