from typing import Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from .mit import MultiframeIntegrationTransformer, ResidualAttentionBlock
from .prompt import AnimalSpecificPrompt
from .cct import CrossFrameCommunicationTransformer
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer,VisionTransformer
import clip
import math
from functools import partial
from torch.nn.init import trunc_normal_

import utils.weight_init_helper as init_helper
from .attention import MultiScaleBlock
from .batchnorm_helper import get_norm
from .stem_helper import PatchEmbed
from .utils import round_width
from .tools import plot_attention

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

from functools import reduce
import operator

import einops

from .attention import TemporalCrossAttention

from clip.model import QuickGELU, Attention
from .weight_loaders import weight_loader_fn_dict
from .vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging

class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        self.leakrelu = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(output_size)
        self.input_size = input_size
        self.output_size = output_size

    def edges2matrix(self, edges, num1, num2):

        num = num1 + num2
        matrix = torch.zeros(num, num).cuda()
        
        for i in edges:
            matrix[i[0], i[1]] = 1
            
        return matrix

    def forward(self, x, edges, num1, num2):
        b = x.shape[0]
        adjacency_matrix = self.edges2matrix(edges, num1, num2)  # 根据边的连接关系设置邻接矩阵
        adjacency_matrix = adjacency_matrix.expand(b, -1, -1)

        # 计算邻接矩阵和权重的乘积
        x = torch.matmul(adjacency_matrix.to(x.dtype), torch.matmul(x, self.weight))
        if self.output_size <= self.input_size:
            x = self.leakrelu(x)
        x = self.norm(x)
        return x

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(GraphConvolution(input_size, hidden_size))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_size, hidden_size))

        # 输出层
        self.layers.append(GraphConvolution(hidden_size, output_size))

    def forward(self, x, edges, num1, num2):
        for layer in self.layers:
            x = layer(x, edges, num1, num2)
        return x

class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                 prompts_alpha=1e-4,
                 prompts_layers=3,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        self.prompts_generator1 = AnimalSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        self.prompts_generator2 = AnimalSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        self.use_cache=use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.cache_animal_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))
        
        self.graph = GCNModel(embed_dim, 256, embed_dim, 2)
        
        self.initialize_parameters()
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        text = text.to(torch.int)
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape
        
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    def encode_video(self, image, name=''):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)

        cls_features, img_features = self.encode_image(image)
        img_features = self.prompts_visual_ln(img_features)
        img_features = img_features @ self.prompts_visual_proj
        
        cls_features = cls_features.view(b, t, -1)
        img_features = img_features.view(b,t,-1,cls_features.shape[-1])
        video_features = self.mit(cls_features)

        return video_features, img_features
    
    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                data_loader = DataLoader(text, batch_size=50, shuffle=False)
                text_features = torch.zeros([int(text.shape[0]), 512]).cuda()
                start_index = 0
                for data in data_loader:
                    text_feature = self.encode_text(data)
                    end_index = int(start_index + text_feature.size(0))
                    # 将 text_feature 填充到 text_features 中的对应位置
                    text_features[start_index:end_index] = text_feature

                    # 更新起始索引
                    start_index = end_index
                if text_features.shape[0] != 140 and text_features.shape[0] != 12 and text_features.shape[0] != 21 and text_features.shape[0] != 14:
                    text_features = text_features.view(int(text_features.shape[0]/50), 50, 512)
                    text_features, _ = torch.max(text_features, dim = 1)
                print('text_features', text_features.shape)
                self.cache_text_features = text_features
        self.train()
        return self.cache_text_features
    
    def cache_animal_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_animal_text_features is None:
                data_loader = DataLoader(text, batch_size=50, shuffle=False)
                text_features = torch.zeros([int(text.shape[0]), 512]).cuda()
                start_index = 0
                for data in data_loader:
                    text_feature = self.encode_text(data)
                    end_index = int(start_index + text_feature.size(0))
                    # 将 text_feature 填充到 text_features 中的对应位置
                    text_features[start_index:end_index] = text_feature

                    # 更新起始索引
                    start_index = end_index
                if text_features.shape[0] != 897 and text_features.shape[0] != 11 and text_features.shape[0] != 173:
                    text_features = text_features.view(int(text_features.shape[0]/50), 50, 512)
                    text_features, _ = torch.max(text_features, dim = 1)
                print('animal_text_features', text_features.shape)
                self.cache_animal_text_features = text_features
        self.train()
        return self.cache_animal_text_features
    
    def cache_animal_logits(self, text, image_features, animal_text_features):
        self.eval()
        with torch.no_grad():
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            animal_text_features = animal_text_features / animal_text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            animal_logits = torch.einsum("bd,kd->bk", image_features, logit_scale * animal_text_features).softmax(dim=-1)
        self.train()
        return animal_logits

    def forward(self, image, text, animal_text, animal_pred, edges, name='',pred=False):
        b = image.shape[0]
        T = image.shape[1]
        video_features, img_features = self.encode_video(image, name)
        if text.shape[0] % 50 == 0:
            num2 = int(text.shape[0] / 50)
            num1 = int(animal_text.shape[0] /50)
        else:
            num2 = int(text.shape[0])
            num1 = int(animal_text.shape[0])
        logit_scale = self.logit_scale.exp()
        
        if self.use_cache:
            if len(text.shape) == 3:
                A, B, _ = text.shape
                text = text.view(A*B, -1)
            else:
                pass
            text_features = self.cache_text(text)
            if len(animal_text.shape) == 3:
                A, B, _ = animal_text.shape
                animal_text = animal_text.view(A*B, -1)
            else:
                pass
            animal_text_features = self.cache_animal_text(animal_text)
        else:
            if len(text.shape) == 3:
                A, B, _ = text.shape
                text = text.view(A*B, -1)
            else:
                pass
            text_features = self.encode_text(text)
            if len(animal_text.shape) == 3:
                A, B, _ = animal_text.shape
                animal_text = animal_text.view(A*B, -1)
            else:
                pass
            animal_text_features = self.encode_text(animal_text)
        
        animal_text_features = animal_text_features.unsqueeze(0).expand(b, -1, -1) # 16, 897, 512
        text_features = text_features.unsqueeze(0).expand(b, -1, -1) # 16, 140, 512
        
        animal_pred = animal_pred.unsqueeze(1) # 16, 1, 512
        
        animal_text_feature = torch.einsum('bac, bcm->bam', [animal_pred.to(animal_text_features.dtype), animal_text_features])
        
        animal_pred = animal_pred.squeeze(1).unsqueeze(-1) # 16, 897
        animal_text_features = torch.mul(animal_pred.to(animal_text_features.dtype), animal_text_features)
        
        edges = edges.clone().detach()

        x = torch.cat([animal_text_features, text_features], dim=1)
        text_features = self.graph(x, edges, num1, num2)[:,animal_text_features.shape[1]:,:] # 2, 140, 512
        
        text_features = text_features + self.prompts_generator1(text_features, animal_text_feature) # 16, 140, 512     16, 1, 512
        
        video_features = video_features.unsqueeze(1)
        video_features = video_features + self.prompts_generator2(video_features, animal_text_feature)
        video_features = video_features.squeeze(1)
        
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)
        return logits, video_features


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, mit_layers=4, pred=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)
   
    return model.eval()


def xclip_load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, 
         use_cache=True, pred=False, prompts_alpha=1e-1, prompts_layers=3, mit_layers=1,
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, 
                        logger=logger,
                        prompts_alpha=prompts_alpha, 
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        pred=pred,
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()


_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]

class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.DATA.NUM_CLASSES,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[1][2],
                ],
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x

class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        
        self.head = head_helper.ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.DATA.NUM_CLASSES,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.NUM_FRAMES // pool_size[0][0],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][2],
                ]
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x

class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        spat_sz = int(math.ceil(cfg.DATA.INPUT_SIZE / 32.0))
        self.head = head_helper.X3DHead(
            dim_in=dim_out,
            dim_inner=dim_inner,
            dim_out=cfg.X3D.DIM_C5,
            num_classes=cfg.DATA.NUM_CLASSES,
            pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            bn_lin5_on=cfg.X3D.BN_LIN5,
        )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x

class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        # assert cfg.DATA.INPUT_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        # Prepare input.
        spatial_size = cfg.DATA.INPUT_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.DATA.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        # num_patches = math.prod(self.patch_dims)
        num_patches = reduce(operator.mul, self.patch_dims, 1)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        pool_q = cfg.MVIT.POOL_Q_KERNEL
        pool_kv = cfg.MVIT.POOL_KV_KERNEL
        pool_skip = cfg.MVIT.POOL_SKIP_KERNEL
        stride_q = cfg.MVIT.POOL_Q_STRIDE
        stride_kv = cfg.MVIT.POOL_KV_STRIDE
        stride_skip = cfg.MVIT.POOL_SKIP_STRIDE

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)

        if len(cfg.MVIT.DIM_MUL) > 1:
            for k in cfg.MVIT.DIM_MUL:
                dim_mul[k[0]] = k[1]
        if len(cfg.MVIT.HEAD_MUL) > 1:
            for k in cfg.MVIT.HEAD_MUL:
                head_mul[k[0]] = k[1]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )

            self.blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    kernel_skip=pool_skip[i] if len(pool_skip) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    stride_skip=stride_skip[i] if len(stride_skip) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                )
            )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.head = head_helper.TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x):
        x = x[0].squeeze(2)
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.INPUT_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.INPUT_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            pos_embed_cls = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed_cls
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x

class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dropout ) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class VideoPrompt(torch.nn.Module):
    def __init__(self, config, actionlist, actiondict, actiontoken, device='cuda'):
        super(VideoPrompt, self).__init__()

        self.device = device
        self.clipmodel, _ = clip.load(config.MODEL.ARCH, device=self.device, jit=False)

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.hidden_size = 512
        self.numF = config.DATA.NUM_FRAMES

        self.prefix = 16
        self.postfix = 16
        self.actionlist = actionlist
        self.actiondict = actiondict
        self.actiontoken = actiontoken
        self.tfm_layers = 1
        self.tfm_heads = 8

        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        self.temporalEmbedding = torch.nn.Embedding(self.numF, self.hidden_size)

        self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers, heads=self.tfm_heads, dropout=self.dropout)

        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)


    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
        self.prompt_actiontoken = torch.zeros(len(actionlist), 77)
        for i, a in enumerate(actionlist):
            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device)
            token = torch.from_numpy(self.actiontoken[a][0])
            self.text_embedding[i][0] = embedding[0]
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)


    def forward(self, vids, inp_actionlist):
        # replace_text_embedding at every iter
        # otherwise RuntimeError: backward through the graph a second time
        self.replace_text_embedding(inp_actionlist)

        # encode text
        tFeature = self.clipmodel.encode_text(self.prompt_actiontoken, self.text_embedding)

        # encode videos
        # vFeature = einops.rearrange(vids.float(), 'b t c -> t b c', t=self.numF)
        
        iFeature = self.clipmodel.encode_image(einops.rearrange(vids, 'b t c h w -> (b t) c h w'))
        vFeature = einops.rearrange(iFeature, '(b t) c -> t b c', t=self.numF)

        # temporal modelling
        tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> t b c', b=vFeature.size(1))
        vFeature = vFeature + tempEmbedding.to(self.device)
        vFeature = self.temporalModelling(vFeature)  
        vFeature = vFeature.mean(dim=0)
        
        vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)
        tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)
        logits = vFeature @ tFeature.t() / 0.07

        return logits

class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(num_layers)]
        )

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1, padding=1, groups=in_feature_dim) for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList(
                [TemporalCrossAttention(spatial_size, in_feature_dim) for _ in range(num_layers)]
            )

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']
            
            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1) # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous() # N, T, L, C
                frame_features += feat
            
            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)
            
            if self.enable_temporal_cross_attention:
                frame_features += self.cross_attention[i](in_features[i]['q'], in_features[i]['k'])

            frame_features = frame_features.flatten(1, 2) # N, T * L, C
            
            x = self.decoder_layers[i](x, frame_features)
        
        return x


class EVLTransformer(nn.Module):

    def __init__(
        self, config,
        backbone_type: str = 'clip',
        backbone_path: str = '/home/jingyinuo/.cache/clip/ViT-B-16.pt',
        backbone_mode: str = 'freeze_fp16',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        cls_dropout: float = 0.5,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_config = self._create_backbone(config, backbone_type, backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoder(
            num_frames=config.DATA.NUM_FRAMES,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(backbone_feature_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(backbone_feature_dim, config.DATA.NUM_CLASSES),
        )


    def _create_backbone(
        self, config,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(return_all_features=True, **vit_presets[config.MODEL.ARCH])
        backbone.load_state_dict(state_dict, strict=True) # weight_loader_fn is expected to strip unused parameters

        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone] # avoid backbone parameter registration

        return vit_presets[config.MODEL.ARCH]


    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # finetune bakbone
            return self.backbone


    def forward(self, x: torch.Tensor):
        x = x[0]
        backbone = self._get_backbone(x)

        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features)
        
        vfeature = x.mean(0)
        
        x = self.proj(x[:, 0, :])

        return x #, vfeature
    
class TimeSformerCLIPInitVideoGuide(nn.Module):
    def __init__(self, class_embed, num_frames):
        super().__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", num_frames=num_frames, ignore_mismatched_sizes=True)
        self.linear1 = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.linear2 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(class_embed)
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone(images)[0]
        x = self.linear1(F.adaptive_avg_pool1d(x.transpose(1, 2), t).transpose(1, 2))
        x = self.pos_encod(x)
        video_features = self.image_model(images.reshape(b*t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)
        query_embed = self.linear2(torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)), 2))
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.group_linear(hs)
        return out