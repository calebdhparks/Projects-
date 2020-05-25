import cv2
import numpy as np
import glob
import pylab as plt
import sys
from PIL import Image
from PIL import ImageFilter
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def Blob(path):
    print("Starting Blob creation for",path)
    folders = glob.glob(path)

    frames_list = []
    for folder in folders:
        for f in sorted(glob.glob(folder + '/*.jpg'), key=numericalSort):
            if ("cir_" not in str(f) and "Trajectory" not in str(f) and "blob" not in str(f)):
                frames_list.append(f)

    gs_image_list = []

    for frame in frames_list:
        image = Image.open(frame)
        image_blur = image.filter(ImageFilter.GaussianBlur(radius=3))
        gs_image = image_blur.convert(mode='L')
        # gs_image.show()
        data = np.asarray(gs_image).astype('float32').flatten()
        data[:250000] = 0
        data[650000:] = 0
        gs_image_list.append(data)

    frame_subtract_list = [gs_image_list[i] - gs_image_list[i - 1] for i in range(len(gs_image_list)) if i > 0]
    # frame_subtract_list = [0 for frame in frame_subtract_list for value in frame if value < 30]
    # frame_subtract_list = [255 for frame in frame_subtract_list for value in frame if value > 30]

    for frame in frame_subtract_list:
        for j in range(len(frame)):
            if frame[j] < 30:
                frame[j] = 0
            else:
                frame[j] = 255

    frame_subtract_list_reshape = [frame_subtract_list[j].reshape((720, 1280)) for j in range(len(frame_subtract_list))]

    ct = 1
    print("Writing Blobs")
    for blob_image in frame_subtract_list_reshape:
        # blob_image[240:480,:420] = 0
        image_no_noise = Image.fromarray(blob_image.astype('uint8'))  # turn into an image
        image_no_noise.save(path+'/blob_image_' + str(ct) + '.jpg')
        ct += 1
    print("Blob.py done")
        # image_no_noise.show()


# np.set_printoptions(threshold=sys.maxsize)

#read in files from folder and create a path to open the folder; place them into a list to access later
