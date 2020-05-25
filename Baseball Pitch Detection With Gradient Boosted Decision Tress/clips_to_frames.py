import os
import json
import string
import random
import subprocess
import multiprocessing
import cv2
import numpy as np
import tqdm

def convert_gif_to_frames(gif):
    
    # Initialize the frame number and create empty frame list
    frame_num = 0
    frame_list = []
    
    # Loop until there are frames left
    while True:
        try:
            # Try to read a frame. Okay is a BOOL if there are frames or not
            okay, frame = gif.read()
            # Append to empty frame list
            frame_list.append(frame)
            # Break if there are no other frames to read
            if not okay:
                break
            # Increment value of the frame number by 1
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break
    return frame_list

def save_frames(frames, path):
    for i in tqdm.trange(len(frames)):
        cv2.imwrite(path[:-4]+"_"+str(i)+".jpg", frames[i])


input_root = '../MLSPdata/clips/'
#input_root = '../MLSPdata/clips/' # Harold's computer

input_clipname_list = []

clips = os.listdir(input_root)
for filename in clips:
    if(filename != ".DS_Store"):
        input_clipname_list.append(filename)

#input_clipname_list = ["RHlEdXq2DuI_14", ] # folder name

for input_clipname in input_clipname_list:
    input_folder = input_root + input_clipname

    output_root = '../MLSPdata/frames/'
    #output_root = '../MLSPdata/frames/' # Harold's computer
    output_path = output_root + input_clipname + '/'
    output_folder = os.path.exists(output_path)
    print("output: "+output_path)
    if not output_folder:
        os.makedirs(output_path)


    files = os.listdir(input_folder)
    for filename in files:
        if not os.path.isdir(filename):
            if filename[-4:] != ".mp4":
                continue
            
            f = input_folder+"/"+filename
            print("source: " + f)
            clip = cv2.VideoCapture(f)
            frame_list = convert_gif_to_frames(clip)
            print("frame length: %d, each frame: %d x %d" % (len(frame_list), len(frame_list[0]), len(frame_list[0][1])))
            print(output_path+filename)
            save_frames(frame_list, output_path+filename)



