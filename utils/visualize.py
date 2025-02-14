import cv2

# video保存为的视频
# encode编码器 'XVID' 'DIVX' 'MJPG' 'X264' 'mp4v'
# fps帧率 ，即每秒多少帧
# size大小 (1280,960) 
# iscolor是否彩色 True False

def visualize(video, video_read, pred, label):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24
    size = (640,360)
    iscolor = True
    writer = cv2.VideoWriter(video, fourcc, fps, size, iscolor)



    #读取视频video_read的每一帧
    capture = cv2.VideoCapture(video_read)
    if capture.isOpened():
        while True:
            #frame为读取到的每一视频帧
            ret, frame = capture.read()
            if ret == False:
                break

            #在视频帧上添加文本text，各参数如下：
            p = 'pred:' + pred
            loc = (20,50)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_size = 0.4
            font_color = (255,255,255)
            font_bold = 1

            l = 'label:' + label
            loc1 = (20,70)

            frame_texted = frame.copy()

            cv2.putText(frame_texted,p,loc,font,font_size,font_color,font_bold)
            cv2.putText(frame_texted,l,loc1,font,font_size,font_color,font_bold)

            #将该帧写入视频video
            writer.write(frame_texted)

    capture.release()
    writer.release()