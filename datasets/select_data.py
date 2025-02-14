import os
import csv
import random
path = '/mnt/sdb/data/jingyinuo/LoTE-Animal'
dir_list = os.listdir(path)

animal_map = {}
k = 0
for dir in dir_list:
    if dir != 'Action.zip' and dir != 'annotation':
        video_list = os.listdir(path+'/'+dir)
        for video in video_list:
            for i in range(len(video)):
                if video[i].isdigit():
                    animal = video[0:i]
                    if animal not in animal_map:
                        animal_map.update({animal: k})
                        k = k + 1
                    break
print(animal_map)

f3 = open('animal_label.csv','w',newline='')
writer3 = csv.writer(f3)
for key in animal_map.keys():
    writer3.writerow([animal_map[key], key])
f3.close()

label_map = {}
i = 0
f1 = open('LoTE_train.csv','w', newline='')
writer1 = csv.writer(f1)
f2 = open('LoTE_test.csv','w', newline='')
writer2 = csv.writer(f2)
for dir in dir_list:
    if dir != 'Action.zip' and dir != 'annotation':
        key = dir
        label_map[key] = i
        label_map.update({key: i})
        i = i + 1
for dir in dir_list:
    if dir != 'Action.zip' and dir != 'annotation':
        length = len(os.listdir(path+'/'+dir))
        print(length)
        video = os.listdir(path+'/'+dir)
        train_idx = random.sample(range(0,length), round(0.7*length))
        test_idx = list(range(length))
        print(len(train_idx))
        for idx in train_idx:
            for i in range(len(video[idx])):
                if video[idx][i].isdigit():
                    animal = video[idx][0:i]
                    print(animal)
                    break
            writer1.writerow([dir+'/'+video[idx], label_map[dir], animal_map[animal]])
            test_idx.remove(idx)
        for idx in test_idx:
            for i in range(len(video[idx])):
                if video[idx][i].isdigit():
                    animal = video[idx][0:i]
                    break
            writer2.writerow([dir+'/'+video[idx], label_map[dir], animal_map[animal]])
f1.close()
f2.close()