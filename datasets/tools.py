import torch

def splitstr(aa, labels):
    '''
    input:
    aa: 从CSV文件中读取的一条原始数据
    labels: 从label.csv文件中读取的数据
    output:
    c: 按照list格式整理的可以load的数据
    '''
    # print(aa)
    a = aa.strip("[(").strip(")]").split("), (")
    c = []
    for b in a :
        d = b.strip("'").strip("'").split("', '")
        for label in labels:
            if d[1] == label[1]:
                d[1] = int(label[0])
        c.append(d)
    return c

def compute_freq(file, dataset):
    freq = {}
    animal_freq = {}
    label_freq = {}
    if dataset == 'AK':
        with open(file, 'r') as fin:
            for line in fin:
                # print(line.strip())
                line_split = line.strip().split("	",1)
                print(line_split)
                animal_label = line_split[1].strip("[(").strip(")]").split("), (")
                for i in animal_label:
                    animal = i.split(',')[1]
                    label = i.split(',')[0]
                    if i not in freq.keys():
                        key = i
                        value = 1
                        freq[key] = value
                        freq.update({key: value})
                    else:
                        freq[i] += 1
                    if animal not in animal_freq.keys():
                        animal_key = animal
                        animal_value = 1
                        animal_freq[animal_key] = animal_value
                        animal_freq.update({animal_key: animal_value})
                    else:
                        animal_freq[animal] += 1
                    if label not in label_freq.keys():
                        label_key = label
                        label_value = 1
                        label_freq[label_key] = label_value
                        label_freq.update({label_key: label_value})
                    else:
                        label_freq[label] += 1
    if dataset == 'mmnet':
        with open(file, 'r') as fin:
            for line in fin:
                # print(line.strip())
                label = line.strip().split(" ")[1]
                animal = line.strip().split(" ")[2]
                label_animal = str(label)+","+str(animal)
                if label_animal not in freq.keys():
                    key = label_animal
                    value = 1
                    freq[key] = value
                    freq.update({key: value})
                else:
                    freq[label_animal] += 1
                if animal not in animal_freq.keys():
                    animal_key = animal
                    animal_value = 1
                    animal_freq[animal_key] = animal_value
                    animal_freq.update({animal_key: animal_value})
                else:
                    animal_freq[animal] += 1
                if label not in label_freq.keys():
                    label_key = label
                    label_value = 1
                    label_freq[label_key] = label_value
                    label_freq.update({label_key: label_value})
                else:
                    label_freq[label] += 1
    return freq, animal_freq, label_freq

# import pandas as pd
# label_file = '/mnt/sdb/data/jingyinuo/animal_kingdom/label.csv'
# classes_all = pd.read_csv(label_file).values.tolist()
# freq, animal_freq, label_freq = compute_freq('/mnt/sdb/data/jingyinuo/mmnet/annotation/composition/train.txt', 'mmnet')

# import csv

# # f = open('data.csv', 'w', newline='')

# # writer = csv.writer(f)

# # for key in freq.keys():
# #     writer.writerow([key, freq[key], label_freq[key.split(',')[0]], animal_freq[key.split(',')[1]]])
# # f.close()

import csv

f = open('description_ak.csv','w', newline='')

writer = csv.writer(f)

# for animal in animal_freq.keys():
#     label = []
#     for key in freq.keys():
#         if animal == key.split(',')[1]:
#             if key.split(',')[0] not in label:
#                 label.append(key.split(',')[1])
#     writer.writerow([animal, animal_freq[animal], len(label)])
# f.close()

with open('/mnt/sdb/data/jingyinuo/code/Video-QA/vicuna/fastchat/serve/action_ak.txt', 'r') as file:
    lines = file.readlines()
for line in lines:
    if line[:6] == 'Action':
        action = line[8:-2]
    elif line[:11] == 'Description':
        description = line[13:-1]
        writer.writerow([action, description])
    else:
        pass
f.close()
        

# keys = freq.keys()
# animal_keys = animal_freq.keys()
# hd_idx = [1,2,15,38,40,48,52,67,68,69,78,90,100,102,104,123,128,133]
# md_idx = [5,7,8,10,13,16,25,26,27,32,39,45,46,47,49,51,58,65,80,84,96,97,99,103,105,108,112,114,116,118,120,135]
# ta_idx = [0,3,4,6,9,11,12,14,17,18,19,20,21,22,23,24,28,29,30,31,33,34,35,36,37,41,42,43,44,50,53,54,55,56,57,59,60,61,62,63,64,66,70,71,72,73,74,75,76,77,79,81,82,83,85,86,87,88,89,91,92,93,94,95,98,101,106,107,109,110,111,113,115,117,119,121,122,124,125,126,127,129,130,131,132,134,136,137,138,139]
# hd_freq = []
# md_freq = []
# ta_freq = []
# for i in hd_idx:
#     for j in keys:
#         if classes_all[i][1] in j:
#             if freq[j] / animal_freq[j.split(',')[1]] == 1:
#                 print(j.split(',')[0])
#             hd_freq.append(freq[j] / animal_freq[j.split(',')[1]])
# for i in md_idx:
#     for j in keys:
#         if classes_all[i][1] in j:
#             md_freq.append(freq[j] / animal_freq[j.split(',')[1]])
# for i in ta_idx:
#     for j in keys:
#         if classes_all[i][1] in j:
#             ta_freq.append(freq[j] / animal_freq[j.split(',')[1]])
# import numpy as np
# print(np.mean(hd_freq))
# print(np.mean(md_freq))
# print(np.mean(ta_freq))

def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames.permute(0,2,1,3,4)]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames.permute(0, 2, 1, 3, 4)
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long().cuda(),
        )
        slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list