import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch.nn as nn

def visualize_attention(attn):
    # 计算注意力权重
    attn = F.softmax(attn,dim=-1)
    attn = attn.cpu().numpy()
    attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
    print('ok')
    print(attn)
    # 使用seaborn的heatmap进行可视化
    sns.heatmap(attn.squeeze(), cmap="YlGnBu", annot=True, fmt=".2f")
    print('ok')
    plt.xlabel("Query positions")
    plt.ylabel("Key positions")
    plt.title("Attention Weight Visualization")
    plt.savefig('vis_attn.png')
    # plt.show()

# 可视化注意力权重
# visualize_attention(q, k)

import torch
import torchvision.models as models


import torch

# 指定.pth文件的路径
# model_path = '/mnt/sdb/data/jingyinuo/results/animal_kingdom/lava_nopred_final_3/best.pth'

# # 加载整个模型
# loaded_model = torch.load(model_path)

# attn = loaded_model['model']['prompts_generator2.decoder.2.cross_attn.proj.weight']

# print(loaded_model['model'].keys())

# 打印加载后的模型
# print(loaded_model['model']['prompts_generator2.decoder.2.cross_attn.proj.weight'].shape) 512*512
# print(loaded_model['model']['prompts_generator2.decoder.1.cross_attn.proj.weight'].shape) 512*512

# print(min(min(row) for row in loaded_model['model']['prompts_generator2.decoder.2.cross_attn.proj.weight']))
# visualize_attention(attn)

def plot_attention(img, attention, name):
    attention = attention[:,0,1:-1].squeeze(1).reshape(attention.shape[0],14,14).unsqueeze(0)                 
    attention = nn.functional.interpolate(attention, size = (224, 224), mode="bilinear").squeeze(0).cpu().numpy() # scale_factor=(1, 16, 16)
    img = np.transpose(img, (0, 2, 3, 1))
    plt.figure(figsize=(10, 80))
    
    # text = ["Original Image", "Head Mean"]
    # for j in range(int(img.shape[0])):
    #     plt.subplot(16, 1, j+1)
    #     plt.imshow(img[j])#设置plt可视化图层为原图
    #     plt.imshow(attention[j],alpha=0.4,cmap='rainbow')#这行将attention图叠加显示，透明度0.4
    # plt.savefig('/data2/jingyinuo/results/vis/attention_map/'+name[0][-12:-4]+'.pdf', dpi=300)
        
    
    text = ["Original Image", "Head Mean"]
    for j in range(int(img.shape[0]/2)):
        for i, fig in enumerate([img[j], attention[j]]):
            plt.subplot(8, 2, j*2+i+1)
            if i == 0:
                plt.imshow(fig, vmin=0, vmax=1)
            if i == 1:
                plt.imshow(fig, cmap='inferno')
            plt.title(text[i])
    plt.savefig('/data2/jingyinuo/results/vis/attention_map/'+name[0][-12:-4]+'.pdf', dpi=300)
    print(name[0]+' saved')
    for j in range(int(img.shape[0]/2), img.shape[0]):
        for i, fig in enumerate([img[j], attention[j]]):
            plt.subplot(8, 2, (j-8)*2+i+1)
            if i == 0:
                plt.imshow(fig, vmin=0, vmax=1)
            if i == 1:
                plt.imshow(fig, cmap='inferno')
            plt.title(text[i])
    plt.savefig('/data2/jingyinuo/results/vis/attention_map/'+name[1][-12:-4]+'.pdf', dpi=300)
    print(name[1]+' saved')
    
    
