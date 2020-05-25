import os
import json
import string
import random
import subprocess
import multiprocessing


def local_clip(filename, start_time, duration, output_filename, output_directory):
    end_time = start_time + duration
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'copy', '-an',
               '-threads', '1',
               '-loglevel', 'panic',
               os.path.join(output_directory,output_filename)]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err.output)
        return err.output

import tqdm
#with open('data/mlb-youtube-segmented.json', 'r') as f:
with open('data/backup_segmented.json', 'r') as f:
    data = json.load(f)
    #pool = multiprocessing.Pool(processes=8)
    #pool.map(wrapper, [data[k] for k in data.keys()])
    clips = [data[k] for k in data.keys()]
    clipnames = [k for k in data.keys()]

    assert(len(clips) == len(clipnames))
    
    videoname_list = ["RHlEdXq2DuI", "nYkRFMXKtU0", "91YyEVUeO8I", "yC7tb1umUqw"]
    type_list = ["changeup", "slider", "curveball"] #"knucklecurve",
    for i in tqdm.trange(len(clips)):
        clipname = clipnames[i]
        clip = clips[i]
        input_directory = '../MLSPdata/videos/'
        output_directory = '../MLSPdata/clips/'
        duration = clip['end']-clip['start']
        videoname = clip['url'].split('=')[-1]
        
        if 'type' in clip.keys():
#            print(clip['type'])
            if clip['type'] not in type_list:
                continue
#            else:
#                print(videoname)

        
        if not os.path.exists(os.path.join(input_directory, videoname+'.mp4')):
#            print("can't find: " + input_directory + videoname + '.mp4')
            continue

        if os.path.exists(output_directory+clipname + "/" + clipname + '.mp4'):
#            print("already exists: " + output_directory+clipname + "/" +clipname + '.mp4')
            continue

        if videoname in videoname_list:
            if not os.path.exists(output_directory + clipname):
                os.makedirs(output_directory + clipname)
#            local_clip(os.path.join(input_directory, filename+'.mp4'), clip['start'], duration, filename+str(cnt)+'_out.mp4', output_directory)
            local_clip(os.path.join(input_directory, videoname+'.mp4'), clip['start'], duration, clipname+'.mp4', output_directory+clipname+"/")

print("Complete!")
