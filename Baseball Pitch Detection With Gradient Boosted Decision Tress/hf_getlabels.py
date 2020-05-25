import os
import json
import string
import random
import subprocess
import multiprocessing
import cv2
import numpy as np
import tqdm


input_root = '../MLSPdata/clips/'
output_root = '../MLSPdata/'
output_file = '../MLSPdata/Labels.txt'

input_clipname_list = []
clips = os.listdir(input_root)
for filename in clips:
    if(filename != ".DS_Store"):
        input_clipname_list.append(filename)

#input_clipname_list = ["RHlEdXq2DuI_14", ] # folder name

label_list = []
slider_list = []
with open(output_file, 'w') as writef:
    with open('data/backup_segmented.json', 'r') as f:
        data = json.load(f)
        clips = [data[k] for k in data.keys()]
        clipnames = [k for k in data.keys()]

        assert(len(clips) == len(clipnames))

        videoname_list = ["RHlEdXq2DuI", "nYkRFMXKtU0", "91YyEVUeO8I", "yC7tb1umUqw"]
        type_list = ["changeup", "slider", "curveball"] #"knucklecurve",
        for i in tqdm.trange(len(clips)):
            clipname = clipnames[i]
            clip = clips[i]
            videoname = clip['url'].split('=')[-1]
            if clipname in input_clipname_list:
#                print(clip)
                if 'type' not in clip.keys() :
                    print(clipname)
                    continue
                if clip['type'] == 'slider':
                    slider_list.append(clipname)
                label_list.append(clip['type'])
                writef.write(clipname +  " " + clip['type'] + '\n')

slider_list = sorted(slider_list)
print(slider_list)
from collections import Counter
print(Counter(label_list))



