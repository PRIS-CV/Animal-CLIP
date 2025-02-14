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
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, all_gather, get_map, get_animal, compute_F1, lt_map
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

import train
import val

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


writer = SummaryWriter()

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--dataset', type=str, default='AK')
    parser.add_argument('--description', type=str, default=None)
    parser.add_argument('--animal_description', type=str, default=None)
    parser.add_argument('--pred', action='store_true')

    args = parser.parse_args()

    config = get_config(args)

    return args, config




def main(config): 
    ##########  DataLoader  ##########
    print('Building DataLoader')
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    ##########  ModelBuilder  ##########
    print('Building ModelBuilder')
    if config.MODEL.MODEL_NAME == 'XCLIP':
        model, _ = model_builder.xclip_load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                            device="cpu", jit=False, 
                            T=config.DATA.NUM_FRAMES, 
                            droppath=config.MODEL.DROP_PATH_RATE, 
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                            use_cache=config.MODEL.FIX_TEXT,
                            pred=config.PRED,
                            logger=logger,
                            )
    elif config.MODEL.MODEL_NAME == 'I3D':
        model = model_builder.ResNet(config)
    elif config.MODEL.MODEL_NAME == 'SLOWFAST':
        model = model_builder.SlowFast(config)
    elif config.MODEL.MODEL_NAME == 'X3D':
        model = model_builder.X3D(config)
    elif config.MODEL.MODEL_NAME == 'MViT':
        model = model_builder.MViT(config)
    elif config.MODEL.MODEL_NAME == 'VideoPrompt':
        actionlist, actiondict, actiontoken = text_prompt(config.DATA.LABEL_LIST, clipbackbone=config.MODEL.ARCH)
        model = model_builder.VideoPrompt(config, actionlist, actiondict, actiontoken)
    elif config.MODEL.MODEL_NAME == 'EVL':
        model = model_builder.EVLTransformer(config)
    
    model = model.cuda()
    
    ##########  Optimizer and Lr_scheduler  ##########
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    
    ##########  LossFunction  ##########
    if config.TRAIN.LOSS == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    if config.TRAIN.LOSS == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    
    if config.TRAIN.LOSS == 'dbl':
        freq_file = ''
        criterion = ResampleLoss(
                    use_sigmoid=True,
                    reweight_func='rebalance',
                    focal=dict(focal=True, balance_param=2.0, gamma=2),
                    logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                    loss_weight=1.0, freq_file=freq_file
                )
        
    if config.TRAIN.LOSS == 'asl':
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)
    
    ##########  DataAugmentation  ##########
    mixup_fn = None
    if config.AUG.MIXUP > 0:
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                        smoothing=config.AUG.LABEL_SMOOTH,
                                        mixup_alpha=config.AUG.MIXUP,
                                        cutmix_alpha=config.AUG.CUTMIX,
                                        switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        mixup_fn = LabelSmoothing(num_classes=config.DATA.NUM_CLASSES,
                                    smoothing=config.AUG.LABEL_SMOOTH) 

    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, config.MODEL.RESUME, model, config.PRED, optimizer, lr_scheduler, logger, inflation=False, convert_from_caffe2=config.MODEL.CONVERT_FROM_CAFFE2)
    if config.DATA.description:
        print("########## using descrition ##########")
        text_labels = generate_text(pd.read_csv(config.DATA.description).values.tolist()) #[140,77]
    else:
        text_labels = generate_text(pd.read_csv(config.DATA.LABEL_LIST).values.tolist()) #[140,77]
    print(text_labels.shape)
    if config.DATA.animal_description:
        print("########## using animal descrition ##########")
        animal_labels = generate_text(pd.read_csv(config.DATA.animal_description).values.tolist())
    else:
        animal_labels = generate_text(pd.read_csv(config.DATA.ANIMAL_LABEL_LIST).values.tolist())
    print(animal_labels.shape)
    if config.TEST.ONLY_TEST:
        map, acc1  = val.validate(val_loader, val_data, text_labels, animal_labels, model, config, logger, vis=False)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {map:.4f} {acc1:.4f}")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train.train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, animal_labels, config, mixup_fn, train_data, logger)

        map, acc1  = val.validate(val_loader, val_data, text_labels, animal_labels, model, config, logger, vis=False)
    
        writer.add_scalar('map', map, epoch)
        writer.add_scalar('acc1', acc1, epoch)
        
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: map: {map:.1f} acc1: {acc1:.4f}")
        if config.DATA.MULTI_CLASSES == True:
            is_best = map > max_accuracy
            max_accuracy = max(max_accuracy, map)
            logger.info(f'MAP max accuracy: {max_accuracy:.2f}')
        elif config.DATA.MULTI_CLASSES == False:
            is_best = acc1 > max_accuracy
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Acc1 max accuracy: {max_accuracy:.2f}')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    if config.DATA.FEATURE_ROOT != None:
        map, acc1  = val.validate_feature_data(val_loader, val_data, text_labels, animal_labels, model, config, logger, vis=False)
    else:
        map, acc1  = val.validate(val_loader, val_data, text_labels, animal_labels, model, config, logger, vis=False)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {map:.4f} {acc1:.4f}")

if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])
    # torch.distributed.barrier(device_ids=[2,3])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True) 
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
