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
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, all_gather, get_map, get_animal, compute_F1, lt_map, compute_precision_recall, compute_average_precision, get_relation, lt_acc, plot_tsne, openai_imagenet_template
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

import torch.nn.functional as F

import pandas as pd

import json

import open_clip

from PIL import Image

writer = SummaryWriter()

@torch.no_grad()
def validate(val_loader, val_data, text_labels, animal_labels, model, config, logger, vis=False, noi=True):
    model.eval()
    
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    map_meter = AverageMeter()
    ani_map_meter = AverageMeter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from torchmetrics.classification import MultilabelAccuracy
    eval_metric = MultilabelAccuracy(num_labels=140, average='micro').to(device)
    
    if config.MODEL.MODEL_NAME == 'XCLIP':
        clip_model, _ = clip.load('ViT-B/16', device)
        # resnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        edges = get_relation(config.DATA.LABEL_LIST, config.DATA.RELATION_FILE)
        edges = torch.tensor(edges).cuda(non_blocking=True)
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            animal_gt = batch_data["animal"]
            label_id = label_id.reshape(-1)
            filename = batch_data['filename']
            
            if config.MODEL.MODEL_NAME == 'XCLIP':
                animal_pred = batch_data["animal"]

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w) # 2, 12, 8, 3, 224, 224
            
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True) 

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                if config.MODEL.MODEL_NAME == 'XCLIP':
                    animal_labels = animal_labels.cuda(non_blocking=True)
                    animal_pred = animal_pred.cuda(non_blocking=True)
                    
                    animal_classes = val_data.animal_classes
                    
                    output, vf = model(image_input, text_inputs, animal_labels, animal_pred, edges, filename, config.PRED) # + output
                    ani_map_meter.update_predictions(animal_pred, animal_gt)
                    
                elif config.MODEL.MODEL_NAME == 'VideoPrompt':
                # for id in action_id:
                    name_map = pd.read_csv(config.DATA.LABEL_LIST).values.tolist()
                    inp_actionlist = [name_map[i][1] for i in range(len(name_map))]
                    output = model(image_input, inp_actionlist)
                else:
                    image_input = pack_pathway_output(config, image_input)
                    output = model(image_input)
        
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity
            
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            
            # 标签序号
            label_id = label_id.reshape((b,-1))
            
            label_real = []
            bb = []
            label = torch.nonzero(label_id)
            for i in range(b):
                for line in label:
                    if line[0] == i:
                        bb.append(line[1])
                label_real.append(bb)
                bb = []
        
            acc1, acc5 = 0, 0
            if config.DATA.MULTI_CLASSES == False:
                label_real = torch.tensor(label_real).cuda(non_blocking=True)
                for i in range(b):
                    if indices_1[i] == label_real[i]:
                        acc1 += 1
                    if label_real[i] in indices_5[i]:
                        acc5 += 1
            acc1_meter.update(float(acc1) / b, b)
            acc5_meter.update(float(acc5) / b, b)
            
            if noi == True:
                rela = {}
                fin = open(config.DATA.RELATION_FILE, 'r')
                for line in fin:
                    animal, labels, num = line.strip().split("	")
                    labels = eval(labels)
                    rela[animal] = labels
                for i in range(b):
                    animals = torch.nonzero(animal_gt[i])
                    if len(animals) == 1:
                        nu = nu + 1
                        for j in indices_5[i]:
                            if j not in rela[animal_classes[animals][1]]:
                                tot = tot + 1
            # 结果可视化
            if vis == True:
                print("######## vis start ########")
                classes = val_data.classes
                for i in range(len(filename)):
                    label = ''
                    video = '/mnt/sdb/data/jingyinuo/results/animal_kingdom/vis/animal/' + filename[i][45:]
                    print(video)
                    video_read = filename[i]
                    for lab in label_real[i]:
                        label = label + '  ' + classes[int(lab)][1]
                    
                    values_5, indices_5 = tot_similarity[i].topk(5, dim=-1)
                    pred = ''
                    for j in range(len(indices_5)):
                        pred = pred + '  ' + classes[indices_5[j]][1]
                    visualize(video, video_read, pred, label)
                    
                print("######## vis end ########")
            
            
            tot_similarity, label_id = all_gather([tot_similarity, label_id])
            map_meter.update_predictions(tot_similarity / n, label_id)
            map, aps = get_map(torch.cat(map_meter.all_preds).cpu().numpy(), torch.cat(map_meter.all_labels).cpu().numpy())
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'mAP: {map:4f}\t'
                )
                
    acc1_meter.sync()
    acc5_meter.sync()
    # map_meter.sync()
    
    ani_map, _ = get_map(torch.cat(ani_map_meter.all_preds).cpu().numpy(), torch.cat(ani_map_meter.all_labels).cpu().numpy())
    print(f'ani_map {ani_map:.4f}')
    
    # 计算mAp, F1@3, F1@5
    map, aps = get_map(torch.cat(map_meter.all_preds).cpu().numpy(), torch.cat(map_meter.all_labels).cpu().numpy())
    pred = torch.cat(map_meter.all_preds).cpu().numpy()
    label = torch.cat(map_meter.all_labels).cpu().numpy()
    f13, p3, r3 = compute_F1(3, pred, label, 'overall')
    f15, p5, r5 = compute_F1(3, pred, label, '')
    
    logger.info(f' * Acc@1 {acc1_meter.avg:.4f} Acc@5 {acc5_meter.avg:.4f}')
    logger.info(f' * f1@3 {f13:.4f} p3 {p3:.4f} r3 {r3:.4f} f1@5 {f15:.4f} p5 {p5:.4f} r5 {r5:.4f} map {map:.4f}')
    
    if config.TEST.TEST_LONG_TAIL == True:
        if config.DATA.MULTI_CLASSES == True:
            hd, md, ta = lt_map(aps)
            logger.info(f' * hd {hd:.4f} md {md:.4f} ta {ta:.4f}')
        else:
            lt_acc1, lt_acc5 = lt_acc(torch.cat(map_meter.all_preds).cpu().numpy(), torch.cat(map_meter.all_labels).cpu().numpy(), config.DATA.DATASET)
            for i in ['hd', 'md', 'tl']:
                if lt_acc1[i+'_num'] != 0:
                    lt_acc1[i] = lt_acc1[i] / lt_acc1[i+'_num']
                    lt_acc5[i] = lt_acc5[i] / lt_acc5[i+'_num']
            logger.info(f'Acc1: * hd {(lt_acc1["hd"]):.4f} md {(lt_acc1["md"]):.4f} ta {(lt_acc1["tl"]):.4f}')
            logger.info(f'Acc5: * hd {(lt_acc5["hd"]):.4f} md {(lt_acc5["md"]):.4f} ta {(lt_acc5["tl"]):.4f}')
    return map, acc1_meter.avg