import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import concurrent.futures
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time

np.set_printoptions(threshold=sys.maxsize)


def passToImage(V, w, h):
    return np.uint8(np.reshape(V, (w, h)))


def show(V, w, h):
    I = Image.fromarray(passToImage(V, w, h))
    I.show()


class Model(object):
    trainX = []
    trainY = []
    alphas = []
    T = 1
    eigFaces = 1
    pictures = 1
    thresholds = []
    weigths = []

    # for correct weight updates
    def set_X(self, X):
        self.trainX = X
        self.eigFaces = len(self.trainX)
        self.pictures = len(self.trainX[0])

    def set_Y(self, Y):
        self.trainY = Y

    def set_t(self, t):
        self.T = t

    def train(self):
        # expects trainX to be  rows: num of eiegn faces  coulmns: number of pictures
        # trainX[i] will be a vector of the ith eigen weight for every picture
        self.thresholds = []
        self.weigths = np.full(self.eigFaces, 1 / self.eigFaces)
        X_train = self.trainX
        Y_train = self.trainY
        self.alphas = []

        for i in range(self.T):
            # Fit a classifier with the specific weights
            clf = DecisionTreeClassifier(max_depth=1, random_state=1)
            clf.fit(X_train, Y_train, sample_weight=self.weigths)
            self.thresholds.append(clf)
            pred_train_i = clf.predict(X_train)

            miss = [int(x) for x in (pred_train_i != Y_train)]

            miss2 = [x if x == 1 else -1 for x in miss]
            # Error
            err_m = np.matmul(self.weigths, miss)

            # Alpha
            alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
            self.alphas.append(alpha_m)

            self.weigths = np.multiply(self.weigths, np.exp([float(x) * alpha_m for x in miss2]))
            # print(self.weigths)
            self.weigths*=(1/np.sum(self.weigths))


def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # print(overlap)

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


def adaboost_train(X_tr, Y_tr, T):
    newModel = Model()
    newModel.set_t(T)
    newModel.set_X(X_tr)
    newModel.set_Y(Y_tr)
    newModel.train()
    return newModel


def adaboost_pred(Model, X_te):
    N = len(X_te)
    Y = np.full(N, 0)
    preds = []

    # T iterations
    for t in range(0, Model.T):
        weak_learner = Model.thresholds[t]
        Y = Y + Model.alphas[t] * np.asarray(weak_learner.predict(X_te))
        # print(Y)
    preds.append(Y)
    return preds


# 2.1
print("Finding EigenFaces")
size=64
lfw1000=np.full(size**2,0)
for filename in os.listdir('lfw1000'):
    im = Image.open('lfw1000/'+filename)
    im=im.resize((size,size))
    x = np.asarray(im.getdata())  # x is already unraveled and of shape (4096,)
    lfw1000=np.vstack((lfw1000,x))
lfw1000=lfw1000[1:]
u,sigma,v=np.linalg.svd(lfw1000)
# eigen faces
for x in range(0,len(v)):
    v[x]=np.absolute(v[x])
    v[x]*=(255.0/v[x].max())
basis=v[:10]
f=plt.figure()
plt.plot(v[0])
f.savefig("eigenFacePlot.pdf",bbox_inches='tight')
# plt.show()
np.savetxt('eigenface.csv',v[0],delimiter=",")

print("Finding and ploting Mean Error")
meanError=[]
for k in range(1,101):
    k_eigen_faces=v[:k]
    norms=[]
    for face in range(0,len(lfw1000)):
        F=lfw1000[face]
        # show(test)
        # print(np.linalg.pinv(k_eigen_faces).shape,test.shape)
        w=np.matmul(F,np.linalg.pinv(k_eigen_faces))
        # print(w.shape)
        F_hat=np.full(len(k_eigen_faces[0]),0)
        for x in range(0,k):
            F_hat=F_hat+w[x]*k_eigen_faces[x]
        norms.append(np.linalg.norm(F-F_hat)**2)
    normArray=np.asarray(norms)
    meanError.append(np.mean(normArray))
plt.clf()
f=plt.figure()
plt.plot(range(1,101),meanError)
plt.xlabel('K')
plt.ylabel('Mean error')
f.savefig("meanError.pdf",bbox_inches='tight')
print(meanError[99])

# 2.2-2.3
def findErr(TrainX, TrainY, TestX, TestY, T):
    newModel = adaboost_train(TrainX, TrainY, T)
    # preds = adaboost_pred(newModel, TrainX)
    preds=adaboost_pred(newModel,TestX)
    preds = np.asarray(preds)[0]
    # Err = f1_score(TrainY, np.sign(preds), average='weighted')
    Err = f1_score(TestY, np.sign(preds), average='weighted')

    return Err

print("Reading files and training Adaboost")
size = 19
lfw1000 = np.full(size ** 2, 0)
for filename in os.listdir('lfw1000'):
    im = Image.open('lfw1000/' + filename)
    im = im.resize((size, size))
    x = np.asarray(im.getdata())  # x is already unraveled and of shape (4096,)
    lfw1000 = np.vstack((lfw1000, x))
lfw1000 = lfw1000[1:]
u, sigma, v = np.linalg.svd(lfw1000)
for x in range(0, len(v)):
    v[x] = np.absolute(v[x])
    v[x] *= (255.0 / v[x].max())
# eigen faces
#  GENERATE GRAPHS
for numEigFace in [10,20,50]:
    errorRate=[]
    basis = v[:numEigFace]
    # POPULATE TRAIN DATA
    TrainingDataX = np.full(numEigFace, 0)
    for filename in os.listdir('train/face'):
        im = Image.open('train/face/' + filename)
        x = np.asarray(im.getdata())
        w = np.matmul(x, np.linalg.pinv(basis))
        TrainingDataX = np.vstack((TrainingDataX, w))
    TrainingDataX = TrainingDataX[1:]
    NonFaceTrainingX = np.full(numEigFace, 0)
    for filename in os.listdir('train/non-face'):
        im = Image.open('train/non-face/' + filename)
        x = np.asarray(im.getdata())
        w = np.matmul(x, np.linalg.pinv(basis))
        NonFaceTrainingX = np.vstack((NonFaceTrainingX, w))
    NonFaceTrainingX = NonFaceTrainingX[1:]

    TrainX = np.concatenate((TrainingDataX, NonFaceTrainingX), axis=0)
    TrainYFace = np.full(len(TrainingDataX), 1)
    TrainYNon_Face = np.full(len(NonFaceTrainingX), -1)
    TrainY = np.append(TrainYFace, TrainYNon_Face, axis=None)


    # POPULATE TEST DATA
    TestDataX = np.full(numEigFace, 0)
    for filename in os.listdir('test/face'):
        im = Image.open('test/face/' + filename)
        x = np.asarray(im.getdata())
        w = np.matmul(x, np.linalg.pinv(basis))
        TestDataX = np.vstack((TestDataX, w))
    TestDataX = TestDataX[1:]
    NonFaceTestX = np.full(numEigFace, 0)
    for filename in os.listdir('test/non-face'):
        im = Image.open('test/non-face/' + filename)
        x = np.asarray(im.getdata())
        w = np.matmul(x, np.linalg.pinv(basis))
        NonFaceTestX = np.vstack((NonFaceTestX, w))
    NonFaceTestX = NonFaceTestX[1:]
    TestX = np.concatenate((TestDataX, NonFaceTestX), axis=0)
    testY=np.append(np.full(len(TestDataX),1),np.full(len(NonFaceTestX),-1))
    print("Ploting Error using F1 score for "+str(numEigFace)+" EigenFaces")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        Ts = [10,50,100,150,200]
        reults=[executor.submit(findErr,TrainX,TrainY,TestX,testY,T) for T in Ts]
        for f in concurrent.futures.as_completed(reults):
            errorRate.append(f.result())
    plt.clf()
    f=plt.figure()
    plt.title("Testing Error with "+str(numEigFace)+" Eigen Faces")
    # plt.title("Training Error with " + str(numEigFace) + " Eigen Faces")
    plt.plot([10, 50, 100, 150, 200], errorRate)
    plt.xlabel('Number of Weak classifiers')
    plt.ylabel('F1-Score')
    name=str(numEigFace)+"_Errors.pdf"
    f.savefig(name,bbox_inches='tight')

# FACE DETECTOR
#
def findFaces(numEigFace,weakLearners,thresh,PhotoName):
    print("Find Faces in",PhotoName)
    start_time = time.time()
    basis = v[:numEigFace]
    # POPULATE TRAIN DATA
    TrainingDataX = np.full(numEigFace, 0)
    for filename in os.listdir('train/face'):
        im = Image.open('train/face/' + filename)
        x = np.asarray(im.getdata())
        w = np.matmul(x, np.linalg.pinv(basis))
        TrainingDataX = np.vstack((TrainingDataX, w))
    TrainingDataX = TrainingDataX[1:]
    NonFaceTrainingX = np.full(numEigFace, 0)
    for filename in os.listdir('train/non-face'):
        im = Image.open('train/non-face/' + filename)
        x = np.asarray(im.getdata())
        w = np.matmul(x, np.linalg.pinv(basis))
        NonFaceTrainingX = np.vstack((NonFaceTrainingX, w))
    NonFaceTrainingX = NonFaceTrainingX[1:]

    TrainX = np.concatenate((TrainingDataX, NonFaceTrainingX), axis=0)
    TrainYFace = np.full(len(TrainingDataX), 1)
    TrainYNon_Face = np.full(len(NonFaceTrainingX), -1)
    TrainY = np.append(TrainYFace, TrainYNon_Face, axis=None)
    #
    FaceDetectModel = adaboost_train(TrainX, TrainY, weakLearners)
    # scale box by full image scales
    newImage = Image.open('photos/'+PhotoName).convert('LA')
    Realwidth, Realheight = newImage.size
    step=5
    print(Realwidth)
    scoreMap = []
    scales = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    for f in scales:
        im = newImage.resize((int(Realwidth * f), int(Realheight * f)))
        newWidth, newHeight = im.size
        scoreMapScale = []
        for b in range(0, newHeight,step):
            rowScoreMap = []
            for x in range(0, newWidth, step):
                # # left top right bottom
                area = (x, b, x + 19, b + 19)
                subIm = im.crop(area)
                # subIm=subIm.resize((19,19))
                rowScoreMap.append(subIm)
            scoreMapScale.append(rowScoreMap)
        scoreMap.append(scoreMapScale)
    scores = []
    maybeface = []
    positions = []
    for w in range(0, len(scoreMap)):
        for y in range(0, len(scoreMap[w])):
            for p in range(0, len(scoreMap[w][y])):
                data = np.asarray(scoreMap[w][y][p].getdata()).transpose()[0]
                newW = np.matmul(data, np.linalg.pinv(basis))
                pred = adaboost_pred(FaceDetectModel, np.reshape(newW, (1, -1)))
                if pred[0] > thresh:
                    positions.append(list(zip([scales[w]], [p*step], [y*step])))
                    # print(w,y,p,pred[0])
                    maybeface.append(scoreMap[w][y][p])
                scores.append(pred[0])
    print(len(maybeface))
    print(positions[0][0][0])
    ScaledPosistion = []
    Pos=np.full(4,0)
    for e in range(0, len(positions)):
        for d in range(0, len(positions[e])):
            scaler = 1 / float(positions[e][d][0])
            width = np.minimum(int(positions[e][d][1] * scaler),Realwidth)
            height = np.minimum(int(positions[e][d][2] * scaler),Realheight)
            offset = int(19 * scaler)
            OWidth=np.minimum(width+offset,Realwidth)
            OHeight=np.minimum(height+offset,Realheight)
            L=list(zip([width], [height], [OWidth], [OHeight]))
            Pos=np.vstack((Pos,np.asarray(L)))
            ScaledPosistion.append(L)
    print(len(ScaledPosistion))
    Pos=Pos[1:]
    print(np.amax(Pos[:,0]),np.amax(Pos[:,1]),np.amax(Pos[:,2]),np.amax(Pos[:,3]))

    print(np.amax(np.asarray(scores)))

    A=non_max_suppression_slow(Pos,.3)
    # print(A)
    newImage=newImage.convert('RGB')
    for x in range (0,len(A)):
        area=[(A[x][0],A[x][1]),(A[x][2],A[x][3])]
        draw = ImageDraw.Draw(newImage)
        draw.rectangle(area,outline="red")
    newImage.show()
    newImage.save("Bounding_Box_"+PhotoName)
    print("--- %s seconds ---" % (time.time() - start_time),"for Image ",PhotoName)

with concurrent.futures.ProcessPoolExecutor() as executor:
    A=executor.submit(findFaces,50,10,1.5,"Big_3.jpg")
    B=executor.submit(findFaces,50,10,1,"nasa.jpg")
    Results=[A,B]
    for f in concurrent.futures.as_completed(Results):
        f.result()
    C=executor.submit(findFaces,35,10,.8,"Solvay.jpg")
    D=executor.submit(findFaces,35,10,.5,"Beatles.jpg")
    Results=[C,D]
    for f in concurrent.futures.as_completed(Results):
        f.result()
