import os
import json
import string
import random
import subprocess
import tqdm

save_dir = '../MLSPdata/videos/'
videoname_list = ["RHlEdXq2DuI", "nYkRFMXKtU0", "91YyEVUeO8I"]#, "yC7tb1umUqw"]

#with open('data/mlb-youtube-segmented.json', 'r') as f:
with open('data/backup_segmented.json', 'r') as f:
    data = json.load(f)
    cnt = 0
    print(len(data))
    #counter = set()
    cnt = 0

    for entry in data:
        entry = data[entry]
        
#        print(entry)
#        cnt += 1
#        if cnt >= 2:
#            break

        # cnt += 1
        # if cnt == 5:
        #    break
        yturl = entry['url']
        ytid = yturl.split('=')[-1]
        # counter.add(yturl)
        
        if ytid not in videoname_list:
            continue

        if os.path.exists(os.path.join(save_dir, ytid+'.mp4')):
            continue
        #print(yturl)

        cmd = 'youtube-dl -f mp4 '+yturl+' -o '+os.path.join(save_dir + ytid+'.mp4')
        os.system(cmd)

