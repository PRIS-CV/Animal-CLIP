import decord
import numpy as np
from PIL import Image
import os

# 加载视频文件
file_path = '/mnt/sdb/data/jingyinuo/LoTE-Animal/annotation/LoTE_test.txt'
with open(file_path, 'r') as file:
    for line in file:
        video_file = '/mnt/sdb/data/jingyinuo/LoTE-Animal/data/' + line.split(' ')[0]
        # # fold_path = '/mnt/sdb/data/jingyinuo/LoTE-Animal/data'
        # fold_path = '/mnt/sdb/data/jingyinuo/mmnet/trimmed_video'
        # sub_fold = os.listdir(fold_path)
        # for f in sub_fold:
        #     # video_path = os.listdir(os.path.join(fold_path, f))
        #     # for v in video_path:
        #     if f[:-4] not in os.listdir('/data2/jingyinuo/data/mmnet/image/'):
        #         video_file = fold_path + '/' + f
        video = decord.VideoReader(video_file)

        # 获取视频总帧数
        total_frames = len(video)

        for i in range(total_frames):
        # 读取帧
            frame = video[i]
            
            # 将帧转换为 numpy 数组
            frame = frame.asnumpy()
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(frame)
            
            # 保存帧为图片文件，文件名格式为 frame_0001.jpg, frame_0002.jpg, ...
            filename = '/data2/jingyinuo/data/LoTE/image/' + line.split(' ')[0][:-4] + '/' + f'_t{i+1:06d}.jpg'
            os.makedirs('/data2/jingyinuo/data/LoTE/image/' + line.split(' ')[0][:-4] + '/', exist_ok=True)
            # filename = '/data2/jingyinuo/data/mmnet/image/' + f[:-4] + '/' + f[:-4] + f'_t{i+1:06d}.jpg'
            # os.makedirs('/data2/jingyinuo/data/mmnet/image/' + f[:-4] + '/', exist_ok=True)
            pil_image.save(filename)

            print(f'Frame {i+1} extracted.')

        print('Extraction complete.')
                # 提取并保存视频帧
    # else:
    #     pass