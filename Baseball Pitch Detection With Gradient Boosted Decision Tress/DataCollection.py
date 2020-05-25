import os
from circle_detection import run
from blob import Blob
import time
A=[x[0] for x in os.walk("./videos/frames")]
# print(A.index("./videos/frames/RHlEdXq2DuI17_out"),len(A))
# A=A[115:250]
print(A)
errorFile=open("./videos/LostBalls", "w")
errorFile.close()
start=time.time()
counter=1
total=len(A)
for folder in A:
    # Blob(folder)
    run(folder)
    print(counter,"out of",total,".",(counter/total)*100,"% complete")
    counter+=1
print("Ran in",(time.time()-start)/60,"minutes")
# run("./videos/frames/RHlEdXq2DuI66_out")