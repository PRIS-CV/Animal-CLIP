import os
import cv2
for dir in os.listdir('/mnt/sdb/data/jingyinuo/LoTE-Animal'):
    if dir != 'Action.zip' and dir != 'annotation' and dir != 'broken_video':
        for video_path in os.listdir('/mnt/sdb/data/jingyinuo/LoTE-Animal' + '/' + dir):
            print(video_path)
            video = cv2.VideoCapture('/mnt/sdb/data/jingyinuo/LoTE-Animal'+ '/' + dir + '/' + video_path)
            while True:
                try:
                    (grabbed, frame) = video.read()
                except:
                    print("EEEEEEEEEEEEEEEEEEEEEEEError")
                    print(dir)
                    print(video_path)
                    break
                if not grabbed:
                    break
                