import numpy as np
import torch.distributed as dist
import torch
from torchvision.models.resnet import resnet50
from torchvision import transforms
import clip
import os
from sklearn.metrics import average_precision_score
from PIL import Image
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score

from iopath.common.file_io import g_pathmgr
import pickle
from collections import OrderedDict
from .c2_model_loading import get_name_convert_func

import pandas as pd

import json

import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(features, ani_labels, labels, ani_label_map, label_map):
    ''' 
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签 
    '''
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne = TSNE(n_components=2, init='pca', n_iter=5000, perplexity=30, random_state=0, learning_rate=200, early_exaggeration=30)
    
    real_labels = []
 
    with open('/mnt/sdb/data/jingyinuo/animal_kingdom/map.json', 'r') as file:
        data = json.load(file)
    
    # swapped_dict = {v: k for k, v in data.items()}
    
    for l in range(ani_labels.shape[0]):
        real_l = list(np.nonzero(ani_labels[l]))
        pre_label = [key for key, value in data.items() if ani_label_map[real_l[0][0]][1] in value]
        
        real_l = list(np.nonzero(labels[l]))
        lab = label_map[real_l[0][0]][1]
        
        real_labels.append(pre_label[0]+ '_' + lab)
    
    
    unique_list = list(set(real_labels))
    
    class_num = len(unique_list) #要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)    #将特征使用PCA降维至2维
    print('tsne_features的shape:',tsne_features.shape)
    x_min, x_max = tsne_features.min(0), tsne_features.max(0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 提取坐标
    # x = tsne_features[:, 0]
    # y = tsne_features[:, 1]
    # z = tsne_features[:, 2]

    # # 获取唯一的类别名
    # unique_labels = np.unique(real_labels)

    # # 创建颜色映射，确保颜色数量足够
    # colors = plt.cm.get_cmap('tab10', len(unique_labels))

    # # 为每个类别分配颜色
    # label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}

    # # 为每个类别绘制点
    # for label in unique_labels:
    #     idx = np.array(real_labels) == label
    #     ax.scatter(x[idx], y[idx], z[idx], label=label, c=[label_to_color[label]], marker='o')

    # # 设置标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()

    # plt.show()
    
    df = pd.DataFrame()
    df["y"] = real_labels
    df["X"] = tsne_features[:,0]
    df["Y"] = tsne_features[:,1]

    sns.scatterplot(x="X", y="Y", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df).set(title="t-SNE Visualization")
    
    return plt

def lt_acc(preds, labels, dataset):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    https://github.com/facebookresearch/SlowFast/blob/2090f2918ac1ce890fdacd8fda2e590a46d5c734/slowfast/utils/meters.py#L231
    """
    # labels = labels.astype('int')
    print('preds.shape', preds.shape)
    print('preds.shape', labels.shape)
    if dataset == 'mmnet':
        map = {'hd':[2,8], 'md':[1,3,4], 'tl':[0,5,6,7,9,10,11]}
    elif dataset == 'LoTE':
        map = {'hd':[1,3,9], 'md':[4,10,11,12,13,16], 'tl':[0,2,5,6,7,8,14,15,17,18,19,20]}
    acc1 = {'hd':0, 'md':0, 'tl':0, 'hd_num':0, 'md_num':0, 'tl_num':0}
    acc5 = {'hd':0, 'md':0, 'tl':0, 'hd_num':0, 'md_num':0, 'tl_num':0}
    # try:
    for i in range(len(labels)):
        label_id = int(np.nonzero(labels[i])[0])
        for key in map.keys():
            if label_id in map[key]:
                acc1[key+'_num'] += 1
                acc5[key+'_num'] += 1
                indices_1 = np.argsort(-preds[i])[:1]  # 获取前k个最大值的索引（-np_list表示取最大值）
                indices_5 = np.argsort(-preds[i])[:5]
                if int(indices_1) == label_id:
                    acc1[key] += 1
                if label_id in indices_5:
                    acc5[key] += 1
                break
    return acc1, acc5

a_map = {2:0,27:1,40:2,45:3,47:4,67:5,68:6,78:7,100:8,102:9,104:10,108:11,116:12,133:13}
def get_relation(label_path, relation_path):
# with open('/mnt/sdb/data/jingyinuo/animal_kingdom/map.json', 'r') as f:
#     animal_map = json.load(f)
# animal_map_reverse = {value: key for key, values in animal_map.items() for value in values}
    # file_out = open('animal-action-relation2.txt', 'w')
    # 写入内容到文件中
    all_classes = pd.read_csv(label_path).values.tolist()
    
    animal_all = []
    animal_action = {}
    fin = open(relation_path, 'r')
    k = 0
    for line in fin:
        new_labels = []
        k += 1
        animal, labels, num = line.strip().split("	")
        # pre_animal = animal_map_reverse[animal]
        pre_animal = animal
        labels = eval(labels)
        # for label in labels:
        #     if label in a_map.keys():
        #         new_labels.append(a_map[label])
        # file_out.write(animal+'	'+str(new_labels)+'	'+str(len(new_labels))+'\n')
        if pre_animal not in animal_action:
            animal_action.update({pre_animal: labels})
        else:
            set1 = set(animal_action[pre_animal])
            set2 = set(labels)
            merged_set = set1.union(set2)
            merged_list = list(merged_set)
            animal_action[pre_animal] = merged_list
    i = 0
    edges = []
    for animal in animal_action:
        for action in animal_action[animal]:
            edges.append((i,k+action))
            edges.append((k+action,i))
        i += 1
    for k in range(len(animal_action.keys())+len(all_classes)):
        edges.append((k,k))
    return edges


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.all_preds = []
        self.all_labels = []
        
    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


# def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
#     if os.path.isfile(config.MODEL.RESUME): 
#         logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
#         checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
#         load_state_dict = checkpoint['model']

#         msg = model.load_state_dict(load_state_dict, strict=False)
#         logger.info(f"resume model: {msg}")

#         try:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

#             start_epoch = checkpoint['epoch'] + 1
#             max_accuracy = checkpoint['max_accuracy']

#             logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
#             del checkpoint
#             torch.cuda.empty_cache()

#             return start_epoch, max_accuracy
#         except:
#             del checkpoint
#             torch.cuda.empty_cache()
#             return 0, 0.

#     else:
#         logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
#         return 0, 0


def load_checkpoint(
    config,
    path_to_checkpoint,
    model,
    pred = False,
    optimizer=None,
    lr_scheduler=None,
    logger=None,
    inflation=False,
    convert_from_caffe2=False,
    epoch_reset=False,
    clear_name_pattern=(),
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert g_pathmgr.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if hasattr(model, 'module') else model
    if convert_from_caffe2:
        with g_pathmgr.open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
            if converted_key in ms.state_dict():
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
                model_blob_shape = ms.state_dict()[converted_key].shape

                # expand shape dims if they differ (eg for converting linear to conv params)
                if len(c2_blob_shape) < len(model_blob_shape):
                    c2_blob_shape += (1,) * (
                        len(model_blob_shape) - len(c2_blob_shape)
                    )
                    caffe2_checkpoint["blobs"][key] = np.reshape(
                        caffe2_checkpoint["blobs"][key], c2_blob_shape
                    )
                # Load BN stats to Sub-BN.
                if (
                    len(model_blob_shape) == 1
                    and len(c2_blob_shape) == 1
                    and model_blob_shape[0] > c2_blob_shape[0]
                    and model_blob_shape[0] % c2_blob_shape[0] == 0
                ):
                    caffe2_checkpoint["blobs"][key] = np.concatenate(
                        [caffe2_checkpoint["blobs"][key]]
                        * (model_blob_shape[0] // c2_blob_shape[0])
                    )
                    c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

                if c2_blob_shape == tuple(model_blob_shape):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    logger.info(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                else:
                    logger.warn(
                        "!! {}: {} does not match {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                if not any(
                    prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):
                    logger.warn(
                        "!! {}: can not be converted, got {}".format(
                            key, converted_key
                        )
                    )
        diff = set(ms.state_dict()) - set(state_dict)
        diff = {d for d in diff if "num_batches_tracked" not in d}
        if len(diff) > 0:
            logger.warn("Not loaded {}".format(diff))
        ms.load_state_dict(state_dict, strict=False)
        if "epoch" in caffe2_checkpoint['blobs'].keys() and not epoch_reset:
            start_epoch = caffe2_checkpoint['blobs']["epoch"] + 1
            if optimizer:
                for k in caffe2_checkpoint['blobs'].keys():
                    if 'optimizer' in k:
                        optimizer.load_state_dict(caffe2_checkpoint[k])
        else:
            start_epoch = 0
        lr_scheduler.load_state_dict({'lr': caffe2_checkpoint['blobs']['lr']})
        try:
            max_accuracy = caffe2_checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {caffe2_checkpoint['epoch']})")
            
            del caffe2_checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del caffe2_checkpoint
            torch.cuda.empty_cache()
        return 0, 0.
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        with g_pathmgr.open(path_to_checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
            
        print(len(checkpoint["model"].keys()))    
            
        model_state_dict_3d = (
            ms.state_dict()
        )
        if 'model' in checkpoint.keys():
            pass
        else:
            checkpoint.update({'model': checkpoint['model_state']})
        checkpoint["model"] = normal_to_sub_bn(
            checkpoint["model"], model_state_dict_3d, logger
        )
        if inflation:
            # Try to inflate the model.
            inflated_model_dict = inflate_weight(
                checkpoint["model"], model_state_dict_3d
            )
            ms.load_state_dict(inflated_model_dict, strict=False)
        else:
            if clear_name_pattern:
                for item in clear_name_pattern:
                    model_state_dict_new = OrderedDict()
                    for k in checkpoint["model"]:
                        if item in k:
                            k_re = k.replace(item, "")
                            model_state_dict_new[k_re] = checkpoint[
                                "model"
                            ][k]
                            logger.info("renaming: {} -> {}".format(k, k_re))
                        else:
                            model_state_dict_new[k] = checkpoint["model"][
                                k
                            ]
                    checkpoint["model"] = model_state_dict_new

            # if pred == False:
            #     ms.visual2 = None
            pre_train_dict = checkpoint["model"]
            model_dict = ms.state_dict()
            print(len(model_dict.keys()))
            # Match pre-trained weights that have same shape as current model.
            pre_train_dict_match = {
                k: v
                for k, v in pre_train_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            # Weights that do not have match from the pre-trained model.
            not_load_layers = [
                k
                for k in model_dict.keys()
                if k not in pre_train_dict_match.keys()
            ]
            # Log weights that are not loaded with the pre-trained weights.
            if not_load_layers:
                for k in not_load_layers:
                    logger.info("Network weights {} not loaded.".format(k))
            # Load pre-trained weights.
            ms.load_state_dict(pre_train_dict_match, strict=False)
            
            # if pred == True:
            #     print('okkkkkkkkkkkk')
            #     clip_visual_state_dict = {
            #         k[7:]: v for k, v in pre_train_dict.items() if ("visual" in k and "prompt" not in k and "message" not in k)
            #     }
                
            #     msg = ms.visual2.load_state_dict(clip_visual_state_dict,strict=False)
            
            start_epoch = 0

            # Load the optimizer state (commonly not done when fine-tuning)
        if "epoch" in checkpoint.keys() and not epoch_reset:
            start_epoch = checkpoint["epoch"] + 1
            # if optimizer:
            #     print(optimizer.state_dict)
            #     for k in checkpoint.keys():
            #         if 'optimizer' in k:
            #             # optimizer.load_state_dict(checkpoint['param_groups'][632])
            #             optimizer.load_state_dict(checkpoint[k])
            #             break
            
        else:
            start_epoch = 0
        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key
    
def normal_to_sub_bn(checkpoint_sd, model_sd, logger):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.info(
                    "{} {} -> {}".format(
                        key, before_shape, checkpoint_sd[key].shape
                    )
                )
    return checkpoint_sd

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def sliding_window(text, window_size, step_size):
    """
    使用滑动窗口对文本进行处理

    参数:
    - text: 输入文本
    - window_size: 窗口大小
    - step_size: 滑动步长

    返回:
    - windows: 包含滑动窗口的列表
    """
    windows = []
    text_length = len(text)

    for start in range(0, text_length, step_size):
        end = start + window_size
        window = text[start:end]
        windows.append(window)

    return windows

# def split_and_pad(sentence, segment_length, total_segments):
#     """
#     将句子切分为指定数量的段落，每段指定长度，不足部分用0填充。

#     参数:
#     - sentence: 输入句子
#     - segment_length: 每段的长度
#     - total_segments: 总的段落数量

#     返回:
#     - segments: 包含切分和填充后的段落的列表
#     """
#     segments = []
#     for i in range(total_segments):
#         start_idx = i * segment_length
#         end_idx = start_idx + segment_length
#         segment = sentence[start_idx:end_idx].ljust(segment_length, '0')
#         segments.append(segment)
#     return segments

def generate_text(data):
    flag = 0
    # if len(data) % 50 == 0:
    #     flag = 1
    if flag == 1:
        text_aug = f"{{}}"
        all_token = []
        classes = []
        for i,c in data:
            windows = sliding_window(text=c, window_size=77, step_size=77)
            token = torch.zeros((30,77))
            
            if len(windows) > 30:
                rng = 30
            else:
                rng = len(windows)

            for i in range(rng):
                token[i] = clip.tokenize(text_aug.format(windows[i]), context_length=77)
            token = token.unsqueeze(0)
            all_token.append(token)
        classes = torch.cat([i for i in all_token])
    else:
        text_aug = f"{{}}"
        # classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data])
        classes = torch.cat([clip.tokenize(text_aug.format(c), truncate=True) for i, c in data])
        if (classes.shape[0] % 50) == 0:
            classes = classes.reshape(int(classes.shape[0]/50), 50, 77)
    return classes


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    https://github.com/facebookresearch/SlowFast/blob/2090f2918ac1ce890fdacd8fda2e590a46d5c734/slowfast/utils/meters.py#L231
    """
    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    # labels = labels.astype('int')
    aps = [0]
    # try:
    aps = average_precision_score(labels, preds, average=None)
    # except ValueError:
    #     print(
    #         "Average precision requires a sufficient number of samples \
    #         in a batch which are missing in this sample."
    #     )
    mean_ap = np.mean(aps)
    return mean_ap, aps

# def get_animal(image, ):
#     ### 抽取中间一帧 采用CLIP预训练模型得到动物预测结果 ###
    
def get_animal(model, image_input, label, device):
    # Load the model
    # dirpath = image_route + image[idx]
    # image_input = np.transpose(image_input, (0, 3, 1, 2))
    image_input = image_input.to(device)
    # text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in label]).to(device)

    # Calculate features
    with torch.no_grad():
    #     image_features = model.encode_image(image_input)
    #     text_features = model.encode_text(text_inputs)

    # # Pick the top most similar labels for the image
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
        output = model(image_input)
        similarity = output.softmax(dim=-1)
    # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # print(similarity.shape)
    
    values, indices = similarity[0].topk(len(label))

    # Re-arrange the prediction tensor
    new_values = torch.zeros(len(label))
    new_indices = indices.tolist()
    for i in range(len(label)):
        new_values[i] = values[new_indices.index(i)]
    # get_map_preds[idx,:] = new_values

    # Print the result
    # for value, index in zip(values, indices):
        # print(f"{label[index]}: {100 * value.item():.2f}%")
        
    return similarity

    # map = get_map(get_map_preds.numpy(), get_map_labels.numpy())
    # print(map)
    
# def get_animal(model, image_input, label):
#     inp = Image.fromarray(inp.astype('uint8'), 'RGB')
#     inp = transforms.ToTensor()(inp).unsqueeze(0)
#     with torch.no_grad():
#         prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
#     return {labels[i]: float(prediction[i]) for i in range(1000)}

def compute_F1(k,predictions,labels,mode_F1):
    idx = np.argsort(predictions,axis = 1)
    for i in range(predictions.shape[0]):
        predictions[i][idx[i][-k:]]=1
        predictions[i][idx[i][:-k]]=0
        
    if mode_F1 == 'overall':
        print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = np.sum(labels[mask]==1)
        p = TP/np.sum(mask)
        r = TP/np.sum(labels==1)
        f1 = 2*p*r/(p+r)
        
#        p_2,r_2,f1_2=compute_F1_fast0tag(predictions,labels)
    else:
        num_class = predictions.shape[1]
        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r  = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:,idx_cls])
            label = np.squeeze(labels[:,idx_cls])
            if np.sum(label>0)==0:
                continue
            binary_label=np.clip(label,0,1)
            f1[idx_cls] = f1_score(binary_label,prediction)#AP(prediction,label,names)
            p[idx_cls] = precision_score(binary_label,prediction)
            r[idx_cls] = recall_score(binary_label,prediction)
        f1 = np.mean(f1)
        p = np.mean(p)
        r = np.mean(r)
    return f1,p,r

def lt_map(results):
    missing_idx = [4,24,34,36,53,63,70,77,86,89,111,113,126]
    for idx in missing_idx:
        results = np.insert(results,idx,0)
    
    hd_idx = [1,2,15,38,40,48,52,67,68,69,78,90,100,102,104,123,128,133]
    md_idx = [5,7,8,10,13,16,25,26,27,32,39,45,46,47,49,51,58,65,80,84,96,97,99,103,105,108,112,114,116,118,120,135]
    ta_idx = [0,3,4,6,9,11,12,14,17,18,19,20,21,22,23,24,28,29,30,31,33,34,35,36,37,41,42,43,44,50,53,54,55,56,57,59,60,61,62,63,64,66,70,71,72,73,74,75,76,77,79,81,82,83,85,86,87,88,89,91,92,93,94,95,98,101,106,107,109,110,111,113,115,117,119,121,122,124,125,126,127,129,130,131,132,134,136,137,138,139]
    
    
    
    hd = []
    for idx in hd_idx:
        hd.append(results[idx])
    hd_mean = np.mean(hd)
    print(hd_mean)
    
    md = []
    for idx in md_idx:
        md.append(results[idx])
    md_mean = np.mean(md)
    print(md_mean)
    
    ta = []
    for idx in ta_idx:
        ta.append(results[idx])
    new_ta = [i for i in ta if i != 0]
    ta_mean = np.mean(new_ta)
    print(ta_mean)
    
    return hd_mean, md_mean, ta_mean

import numpy as np


def compute_precision_recall(scores, labels, num_gt=None):
    """Compute precision and recall.
    计算只取分数最高的一个、取最高的两个……全部都取的情况下的精确率和召回率

    Args:
      scores: A float numpy array representing detection score
      labels: A boolean numpy array representing true/false positive labels
      num_gt: Number of ground truth instances

    Raises:
      ValueError: if the input is not of the correct format

    Returns:
      precision: Fraction of positive instances over detected ones. This value is
        None if no ground truth labels are present.
      recall: Fraction of detected positive instance over all positive instances.
        This value is None if no ground truth labels are present.

    """
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype != bool
        or len(labels.shape) != 1
    ):
        raise ValueError("labels must be single dimension bool numpy array")

    # 添加
    num_gt = num_gt or np.sum(labels)

    if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
        raise ValueError("scores must be single dimension numpy array")

    if num_gt < np.sum(labels):
        raise ValueError(
            "Number of true positives must be smaller than num_gt."
        )

    if len(scores) != len(labels):
        raise ValueError("scores and labels must be of the same size.")

    if num_gt == 0:
        return None, None

    sorted_indices = np.argsort(scores)  # 按照值升序排序索引
    sorted_indices = sorted_indices[::-1]  # 降序
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (
        cum_true_positives + cum_false_positives
    )
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit.
    计算recall-precision曲线下的面积

    Precision is modified to ensure that it does not decrease as recall
    decrease.

    Args:
      precision: A float [N, 1] numpy array of precisions
      recall: A float [N, 1] numpy array of recalls

    Raises:
      ValueError: if the input is not of the correct format

    Returns:
      average_precison: The area under the precision recall curve. NaN if
        precision and recall are None.

    """
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(
        recall, np.ndarray
    ):
        raise ValueError("precision and recall must be numpy array")
    if precision.dtype != float or recall.dtype != float:
        raise ValueError("input must be float numpy array.")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices]
    )
    return average_precision

openai_imagenet_template = [
    lambda c: f'a photo of {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]