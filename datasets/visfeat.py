import os
import cv2
import torch
import numpy as np 
from PIL import Image
from clip import clip
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import time




def load_clip_cpu(backbone_name):
    model_path = '/home/jingyinuo/.cache/clip/ViT-B-16.pt'
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    model = clip.build_model(state_dict or model.state_dict())

    return model


def transform_center():
    interp_mode = Image.BICUBIC
    tfm_test = []
    tfm_test += [Resize(224, interpolation=interp_mode)] 
    tfm_test += [CenterCrop((224,224))]
    tfm_test += [ToTensor()]
    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    tfm_test += [normalize]
    tfm_test = Compose(tfm_test)

    return tfm_test


def get_videos(vidname, read_path, centrans):
    allframes = []
    videoins = read_path + vidname
    vvv = cv2.VideoCapture(videoins)
    if not vvv.isOpened():
        print('Video is not opened! {}'.format(videoins))
    else:  
        fps = vvv.get(cv2.CAP_PROP_FPS)  
        totalFrameNumber = vvv.get(cv2.CAP_PROP_FRAME_COUNT)
        size = (int(vvv.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vvv.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
        second = totalFrameNumber//fps

        if totalFrameNumber != 0:
            for _ in range(int(totalFrameNumber)):
                rval, frame = vvv.read()   
                if frame is not None: 
                    img = Image.fromarray(frame.astype('uint8')).convert('RGB')
                    imgtrans = centrans(img).numpy()                 
                    allframes.append(imgtrans)  

    return np.array(allframes)

def visfeat(savepath, datapath, ARCH):
    maxlen = 2500                                           # the maximum number of video frames that GPU can process
    os.chdir(datapath)
    allvideos_temp = os.listdir()
    allvideos = []
    
    if os.path.isdir(os.path.join(datapath, allvideos_temp[0])):
        for dir in allvideos_temp:
            path = ['/' + dir + '/' + i for i in os.listdir(os.path.join(datapath, dir))]
            allvideos = allvideos + path
    else:
        allvideos = allvideos_temp

    allvideos.sort()
    centrans = transform_center()
    # load CLIP pre-trained parameters
    device = 'cuda'
    clip_model = load_clip_cpu(ARCH)
    clip_model.to(device)
    for paramclip in clip_model.parameters():
        paramclip.requires_grad = False


    for vid in range(len(allvideos)):
        t1 = time.time()
        vidone = get_videos(allvideos[vid], datapath, centrans)      # shape = (T,3,224,224)image.png
        print(vidone.shape)
        print('transform %d video has been done!' % vid)
        vidinsfeat = []      
        for k in range(int(len(vidone)/maxlen)+1):         # if the video is too long, split the video
            segframes = torch.from_numpy(vidone[k*maxlen:(k+1)*maxlen]).to(device)
            if segframes.shape[0] != 0:
                vis_feats = clip_model.encode_image(segframes)
                vidinsfeat = vidinsfeat + vis_feats.cpu().numpy().tolist()
        vidinsfeat = np.array(vidinsfeat)                  # shape = (T,512)

        assert(len(vidinsfeat) == len(vidone))
        if not os.path.exists(savepath):
                os.mkdir(savepath)
        if os.path.isdir(os.path.join(datapath, allvideos_temp[0])):
            if not os.path.exists(savepath + '/' + allvideos[vid].split('/')[1]):
                os.mkdir(savepath + '/' + allvideos[vid].split('/')[1])
        np.save(savepath+allvideos[vid][:-4]+'.npy', vidinsfeat)
        t2 = time.time()

        print('visual features of %d video have been done!' % vid)
        
        print('total time : %4f' % ((t2-t1)*(30100-vid)))

    print('all %d visual features have been done!' % len(allvideos))
