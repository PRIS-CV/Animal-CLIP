import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.VAL_FILE = ''
_C.DATA.DATASET = 'AK'
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_FRAMES = 8
_C.DATA.NUM_CLASSES = 400
_C.DATA.LABEL_LIST = 'labels/kinetics_400_labels.csv'
_C.DATA.MULTI_CLASSES = False
_C.DATA.CLASSES = None
_C.DATA.NUM_ANIMAL_CLASSES = 896
_C.DATA.ANIMAL_LABEL_LIST = ''
_C.DATA.INPUT_CHANNEL_NUM = []
_C.DATA.FEATURE_ROOT = None
_C.DATA.description = None
_C.DATA.animal_description = None
_C.DATA.RELATION_FILE = ''
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.PRETRAINED = None
_C.MODEL.RESUME = None
_C.MODEL.FIX_TEXT = True
_C.MODEL.FC_INIT_STD = 0.01
_C.MODEL.DROPCONNECT_RATE = 0.0
_C.MODEL.HEAD_ACT = "softmax"
_C.MODEL.DROPOUT_RATE = 0.5
_C.MODEL.MODEL_NAME = 'XCLIP'
_C.MODEL.CONVERT_FROM_CAFFE2 = False

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit", "xclip", 'ViT-B/16', 'ViT-B/32', 'ViT-L/14']

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.WARMUP_EPOCHS = 5.0
_C.TRAIN.WEIGHT_DECAY = 0.001 
_C.TRAIN.WEIGHT_DECAY_STEPS = 2e5
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_POLICY = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.LOSS = 'ce'
_C.TRAIN.LRS = []
_C.TRAIN.STEPS = []

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.TRAIN.BASE_LR_SCALE_NUM_SHARDS = False

# Momentum.
_C.TRAIN.MOMENTUM = 0.9

# Momentum dampening.
_C.TRAIN.DAMPENING = 0.0

# Nesterov momentum.
_C.TRAIN.NESTEROV = True

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.TRAIN.ZERO_WD_1D_PARAM = False

# -----------------------------------------------------------------------------
# ResNet settings
# -----------------------------------------------------------------------------
_C.RESNET = CN()
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CN()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]


# ---------------------------------------------------------------------------- #
# X3D  options
# See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
# ---------------------------------------------------------------------------- #
_C.X3D = CN()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0

# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0

# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048

# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12

# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False

# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False

# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks.
_C.X3D.CHANNELWISE_3x3x3 = True


# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CN()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [2, 4, 4]

# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"

# Dimension multiplication. If 2.0 is used, then the next block will increase the
# dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication. If 2.0 is used, then the next block will increase the
# number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Kernel size for the Pool KV. Format: [[kernel_t_i, kernel_h_i, kernel_w_i], ...,]
_C.MVIT.POOL_KV_KERNEL = [[]]

# Kernel size for the Pool Q. Format: [[kernel_t_i, kernel_h_i, kernel_w_i], ...,]
_C.MVIT.POOL_Q_KERNEL = [[]]

# Kernel size for the Pool Skip. Format: [[kernel_t_i, kernel_h_i, kernel_w_i], ...,]
_C.MVIT.POOL_SKIP_KERNEL = [[]]

# Stride size for the Pool KV. Format: [[stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = [[]]

# Stride size for the Pool Q. Format: [[stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = [[]]

# Stride size for the Pool Skip. Format: [[stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_SKIP_STRIDE = [[]]

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CN()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5

_C.MULTIGRID = CN()
_C.MULTIGRID.SHORT_CYCLE = False


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.1
_C.AUG.COLOR_JITTER = 0.8
_C.AUG.GRAY_SCALE = 0.2
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False
_C.TEST.TEST_LONG_TAIL = True
_C.TEST.TSNE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 50
_C.SEED = 1024
_C.NUM_SHARDS = 1
_C.NUM_WORKERS = 16
_C.PRED = False



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.loss:
        config.TRAIN.LOSS = args.loss
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.output:
        config.OUTPUT = args.output
    if args.only_test:
        config.TEST.ONLY_TEST = True
    if args.description:
        config.DATA.description = args.description
    if args.animal_description:
        config.DATA.animal_description = args.animal_description
    if args.pred:
        config.PRED = args.pred
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

def assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.TRAIN.BASE_LR_SCALE_NUM_SHARDS:
        cfg.TRAIN.BASE_LR *= cfg.NUM_SHARDS
        cfg.TRAIN.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.TRAIN.COSINE_END_LR *= cfg.NUM_SHARDS
    return cfg