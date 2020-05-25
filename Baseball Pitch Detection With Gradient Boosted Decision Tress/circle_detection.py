import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import re
import json


'''This is the command to read in the GIF. You need Videocapture which makes a cv2 Video Object.
Add your GIF file name in the parameter of this function.
'''
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def detect_and_show_circles(input_img, output_img,box):
    # detect circles in the image
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(input_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100,param2=1,maxRadius=10,minRadius=4)
    (xmin,ymin,xmax,ymax)=box
    # ensure at least some circles were found
    ball = (0, 0, 0)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        circles=circles[circles[:, 1].argsort()]
        # print(circles)
        # loop over the (x, y) coordinates and radius of the circles
        foundBall=False
        for (x, y, r) in circles:
        #     # draw the circle in the output image, then draw a rectangle
        #     # corresponding to the center of the circle
            if(x>=xmin and y>=ymin and x<=xmax and y<=ymax and not foundBall):
                # print([x,y,r])
                cv2.circle(output_img, (x, y), r, (0, 255, 0), 4)
                ball=(x,y,r)
                foundBall=True
                # cv2.rectangle(output_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
#         return output_img
#         plt.imshow(np.hstack([input_img, output_img]))
    return output_img,ball

def showArc(img,ballLocal):
    for (x, y, r) in ballLocal:
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        # cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return img

def writeJSON(path,baseballs,frame_count):
    clipName = path.split('/')[-1].split('_')[0]
    labelFile = open("./videos/Videolabels", 'r')
    lines = labelFile.readlines()
    label = "N/A"
    for line in lines:
        clipNameFile = line.split(" ")[0].split('.')[0].split("_")[0]
        # print(clipName,clipNameFile)
        if clipName == clipNameFile:
            label = line.split(" ")[1].strip('\n')
    data = {}
    data['balls'] = []
    data['label'] = label
    data['frames'] = str(frame_count)
    firstX,t,r=baseballs[0]
    if firstX<450:
        hand="L"
    else:
        hand="R"
    data['Hand']=hand
    for i in range(0, baseballs.shape[0]):
        x, y, r = baseballs[i]
        data['balls'].append({
            'X': str(x),
            'Y': str(y),
        })
    output_path="./videos/dataCollection"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path+ "/" + clipName + ".json", 'w') as outfile:
        json.dump(data, outfile)
    print("Wrote",output_path+ "/" + clipName + ".json")

def run(directory):
    ball=(0,0,0)
    lastBall=(0,0,0)
    xmin=300
    xmax=800
    ymin=200
    ymax=480
    box=(xmin,ymin,xmax,ymax)
    firstFrame=True
    baseballs=[]
    img=[]
    missedBall=False
    print("Finding Circles in",directory)
    frame_count=1
    for filename in sorted_aphanumeric(os.listdir(directory)):
        if("cir_" not in str(filename) and "Trajectory" not in str(filename) and "blob" in str(filename)):
            frame_count+=1
            if(firstFrame):
                # print("cir" not in filename)
                im_cv = cv2.imread(directory+"/"+filename)
                img=im_cv
                # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
                output = im_cv.copy()
                out_img,newBall=detect_and_show_circles(im_cv, output,box)
                cv2.imwrite(directory+"/cir_" + filename, out_img)
                ball=newBall
                lastBall=ball
                baseballs.append(ball)
                firstFrame=False
            else:
                # print("cir" not in filename)
                (x,y,r)=ball
                (oldX,oldY,oldR)=lastBall
                if x==0 and y==0 and r==0:
                    # print(lastBall)
                    shift=100
                    xmin = oldX
                    xmax = oldX+shift
                    ymin = oldY-20
                    ymax = oldY+shift
                    box = (xmin, ymin, xmax, ymax)
                    missedBall=True
                else:
                    offset=20
                    xmin = x
                    xmax = x+offset
                    ymin = y-offset
                    ymax = y+offset
                    box = (xmin, ymin, xmax, ymax)
                # print(filename)
                im_cv = cv2.imread(directory + "/" + filename)
                # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
                output = im_cv.copy()
                out_img, newBall = detect_and_show_circles(im_cv, output, box)
                cv2.imwrite(directory + "/cir_" + filename, out_img)
                ball = newBall
                (x,y,r)=ball
                if x!=0 and y!=0:
                    baseballs.append(ball)
                    lastBall=ball
    baseballs=np.array(baseballs)
    TrasjectoryImage=showArc(img,baseballs)
    if (baseballs.size!=0):
        print("Balls Found")
        cv2.imwrite(directory + "/Trajectory.jpg", TrasjectoryImage)
        print("Writing JSON")
        writeJSON(directory,baseballs,frame_count)
    if missedBall:
        print("Wrote too LostBalls file")
        MissesFile = open("./videos/LostBalls", "a")
        MissesFile.write(directory+"\n")
        MissesFile.close()



# gif_ball = cv2.VideoCapture('haha.mp4')
# print(git_ball)

# frame_list = convert_gif_to_frames(gif_ball)
# print(len(frame_list), len(frame_list[0]), len(frame_list[1]))
# im_cv = frame_list[35]
# path="./videos/frames/RHlEdXq2DuI113_out"
# path="./pitch"
# run(path)
# find label

# im_cv = cv2.imread("test.jpeg")
# im_cv = cv2.imread("ball.png")
# im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
# # plt.imshow(im_rgb)
#
# output = im_rgb.copy()
# # plt.imshow(im_rgb)
# detect_and_show_circles(im_rgb, output)

