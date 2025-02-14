import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler, Scheduler
import torch.distributed as dist

import logging
import math
import numpy as np
import torch

def is_main_process():
    return dist.get_rank() == 0

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(name)
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]


def fix_text(model):
    for name, param in model.named_parameters():
        if "visual." in name or "mit" in name or "prompts" in name or 'graph' in name:
            continue
        else:
            param.requires_grad=False

def build_optimizer(config, model):
    model = model.module if hasattr(model, 'module') else model
    if config.MODEL.MODEL_NAME == 'XCLIP':
        
        # fix text
        if config.MODEL.FIX_TEXT:
            fix_text(model)
        
        # set decay and lr
        skip = {}
        skip_keywords = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_keywords = model.no_weight_decay_keywords()
        clip_parameters = set_weight_decay(model, skip, skip_keywords, 
            weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR, 
            have=(), not_have=("prompts", "mit", "message_", 'graph') # ,
        )
        msg_parameters = set_weight_decay(model, skip, skip_keywords,
            weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
            have=("message_",), not_have=()
        )
        mit_parameters = set_weight_decay(model, skip, skip_keywords,
            weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
            have=("mit",), not_have=()
        )
        prompts_parameters = set_weight_decay(model, skip, skip_keywords, 
            weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
            have=("prompts",), not_have=()
        )
        graph_parameters = set_weight_decay(model, skip, skip_keywords, 
            weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10, 
            have=("graph",), not_have=()
        )

        optim_params = clip_parameters + mit_parameters + prompts_parameters + msg_parameters + graph_parameters
        # optimizer = optim.AdamW(clip_parameters + mit_parameters + prompts_parameters + msg_parameters,
        #                     betas=(0.9, 0.98), eps=1e-8,)
    elif config.MODEL.MODEL_NAME == 'VideoPrompt' or 'EVL':
        optim_params = model.parameters()
    
    else:
        bn_parameters = []
        non_bn_parameters = []
        zero_parameters = []
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        for name, m in model.named_modules():
            is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
            for p in m.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if is_bn:
                    bn_parameters.append(p)
                elif name in skip or (
                    (len(p.shape) == 1 or name.endswith(".bias"))
                    and config.TRAIN.ZERO_WD_1D_PARAM
                ):
                    zero_parameters.append(p)
                else:
                    non_bn_parameters.append(p)

        optim_params = [
            {"params": bn_parameters, "weight_decay": config.BN.WEIGHT_DECAY},
            {"params": non_bn_parameters, "weight_decay": config.TRAIN.WEIGHT_DECAY},
            {"params": zero_parameters, "weight_decay": 0.0},
        ]
        optim_params = [x for x in optim_params if len(x["params"])]

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_parameters
        ) + len(
            zero_parameters
        ), "parameter size does not match: {} + {} + {} != {}".format(
            len(non_bn_parameters),
            len(bn_parameters),
            len(zero_parameters),
            len(list(model.parameters())),
        )
        print(
            "bn {}, non bn {}, zero {}".format(
                len(bn_parameters), len(non_bn_parameters), len(zero_parameters)
            )
        )

    if config.TRAIN.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            dampening=config.TRAIN.DAMPENING,
            nesterov=config.TRAIN.NESTEROV,
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=config.TRAIN.LR,
            betas=(0.9, 0.999),
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    elif config.TRAIN.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=config.TRAIN.LR,
            betas=(0.9, 0.98),
            eps=1e-08,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(config.TRAIN.OPTIMIZER)
        )


def build_scheduler(config, optimizer, n_iter_per_epoch):
    if config.TRAIN.LR_POLICY == 'cosine':
        num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
        warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.LR / 100,
            warmup_lr_init=0,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    elif config.TRAIN.LR_POLICY == 'steps_with_relative_lrs':
        lr_scheduler = RelativeLRScheduler(
            optimizer,
            lr = config.TRAIN.LR,
            LRS = config.TRAIN.LRS,
            STEPS = config.TRAIN.STEPS,
            MAX_EPOCH = config.TRAIN.EPOCHS,
            num = n_iter_per_epoch,
        )
    elif config.TRAIN.LR_POLICY == 'cosineannealingwarmrestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  
                                                                            T_0=int(config.TRAIN.WEIGHT_DECAY_STEPS), 
                                                                            eta_min=config.TRAIN.LR*0.01, 
                                                                            last_epoch=-1)
    return lr_scheduler

def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1

_logger = logging.getLogger(__name__)


class RelativeLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: optim.Optimizer,
            lr: 0.1,
            LRS: [0.1, 0.01],
            STEPS: [0, 40, 100],
            MAX_EPOCH: 100,
            num: 100,
    ):
        assert lr >= 0
        self.lr = lr
        self.LRS = LRS
        self.STEPS = STEPS
        self.MAX_EPOCH = MAX_EPOCH
        self.num = num

    def step_update(self, t):
        steps = self.STEPS + [self.MAX_EPOCH]
        for ind, step in enumerate(steps):  # NoQA
            if t/self.num < step:
                break
            index = ind - 1
        
        lr = self.LRS[index] * self.lr
        return lr
        