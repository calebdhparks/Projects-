import os
from hf_circle_detection import run
from hf_blob import Blob
import time
import tqdm

A=[x[0] for x in os.walk("../MLSPdata/frames")]

# print(A.index("./videos/frames/RHlEdXq2DuI17_out"),len(A))
# A=A[115:250]
print(len(A))

start=time.time()

# counter=1
total=len(A)
for i in tqdm.trange(1, len(A)):
    
    folder = A[i]    
    Blob(folder)

    # print(A[i])
    run(folder)
    # print(counter,"out of",total,".",(counter/total)*100,"% complete")
    # counter+=1
print("Ran in",(time.time()-start)/60,"minutes")
# run("./videos/frames/RHlEdXq2DuI66_out")