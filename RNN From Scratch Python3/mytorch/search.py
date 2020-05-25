import numpy as np
import sys
import pdb

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search
    forward_path=""
    maxIdx=np.argmax(y_probs,axis=0)
    maxs=np.max(y_probs,axis=0)
    forward_prob=np.prod(maxs)
    last=maxIdx[0]
    for x in range(1,len(maxIdx)):
        if maxIdx[x]==last:
            maxIdx[x]=0
        else:
            last=maxIdx[x]
    maxIdx = maxIdx[maxIdx > 0] - 1
    forward_path="".join([SymbolSets[y] for y in maxIdx])
    return (forward_path, forward_prob)




##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
def initPaths(Symbols,y):
    #y should be y_probs[:,0,:] 1d
    InitialBlankPathScore = {}
    InitialPathScore = {}
    path=""
    InitialBlankPathScore[path]=y[0]
    InitialPathsWithFinalBlank = set([path])
    InitialPathsWithFinalSymbol = set([])
    for x in range(0,len(Symbols)):
        path=Symbols[x]
        InitialPathScore[path]=y[x+1]
        InitialPathsWithFinalSymbol.add(path)
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol,InitialBlankPathScore, InitialPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    #print("\ninput to prune",PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth)
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    scorelist=[]
    i=0
    for p in PathsWithTerminalBlank:
        scorelist.append((p,BlankPathScore[p]))
        i+=1
    i = 0
    for p in PathsWithTerminalSymbol:
        scorelist.append((p,PathScore[p]))
        i += 1
    scorelist = (sorted(scorelist, key = lambda x: x[1],reverse=True))
    #print(scorelist)
    #find cutoff
    cutoff=[]
    if BeamWidth < len(scorelist):
        cutoff=scorelist[BeamWidth-1]
    else:
        cutoff=scorelist[-1]
    #print("In prune",scorelist,cutoff)
    # pdb.set_trace()
    #print(BeamWidth, scorelist, cutoff,'\n')
    # sys.exit()
    PrunedPathsWithTerminalBlank = set([])
    i=0
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p]>= cutoff[1]:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p]=BlankPathScore[p]
        i+=1

    PrunedPathsWithTerminalSymbol = set([])
    i=0
    for p in PathsWithTerminalSymbol:
        #print(p,PathScore[i],cutoff[1])
        if PathScore[p]>=cutoff[1]:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p]=PathScore[p]
        i+=1
    return PrunedPathsWithTerminalBlank,PrunedPathsWithTerminalSymbol,PrunedBlankPathScore,PrunedPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y,BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set([])
    UpdatedBlankPathScore = {}

    i=0
    for p in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank.add(p)
        UpdatedBlankPathScore[p]=BlankPathScore[p]*y
        i+=1
    i=0
    for p in PathsWithTerminalSymbol:
        if p in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[p]+=PathScore[p]*y
        else:
            UpdatedPathsWithTerminalBlank.add(p)
            UpdatedBlankPathScore[p]=PathScore[p]*y
        i+=1
    #print('b',UpdatedPathsWithTerminalBlank,UpdatedBlankPathScore)
    return UpdatedPathsWithTerminalBlank,UpdatedBlankPathScore

def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y,BlankPathScore, PathScore):
    #print("\nInput to extend",PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y,BlankPathScore, PathScore,"\n")
    UpdatedPathsWithTerminalSymbol = set([])
    UpdatedPathScore = {}
    i=0
    for p in PathsWithTerminalBlank:
        c_i=1
        for c in SymbolSet:
            newpath=p+c
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath]=BlankPathScore[p]*y[c_i]
            c_i+=1
        i+=1
    i=0
    tempPaths=[]
    tempScores=[]
        #print("\n",PathsWithTerminalSymbol,len(PathsWithTerminalSymbol))
    for p in PathsWithTerminalSymbol:
        #print("\n",p)
        c_i=1
        for c in SymbolSet:
            newpath=p
            if p[-1]!=c:
                newpath=p+c
            #print(newpath,UpdatedPathsWithTerminalSymbol,i)
            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath]+=PathScore[p]*y[c_i]
            else:
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath]=PathScore[p]*y[c_i]
            c_i+=1
        i+=1
    #print('s',UpdatedPathsWithTerminalSymbol, UpdatedPathScore)
    return UpdatedPathsWithTerminalSymbol,UpdatedPathScore

def  MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    MergedPaths=PathsWithTerminalSymbol
    FinalPathScore=PathScore
    i=0
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            #indexBlank=list(PathsWithTerminalBlank).index(p)
            FinalPathScore[p]+=BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p]=BlankPathScore[p]
        i+=1
    return MergedPaths, FinalPathScore

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search
    PathScore=[]
    BlankPathScore=[]
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore=initPaths(SymbolSets,y_probs[:,0,:])
    #print(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)
    for t in range(1,y_probs.shape[1]):
        # print("start",t)
        # pdb.set_trace()
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,NewBlankPathScore, NewPathScore, BeamWidth)
        #print("prune", t,PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore)
        # pdb.set_trace()
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank,PathsWithTerminalSymbol, y_probs[0, t,:],BlankPathScore,PathScore)
        # print("blanks", t,NewPathsWithTerminalBlank, NewBlankPathScore)
        # pdb.set_trace()
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:, t,:],BlankPathScore,PathScore)
        # print("symbol", t,NewPathsWithTerminalSymbol, NewPathScore)
        # pdb.set_trace()
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore,NewPathsWithTerminalSymbol, NewPathScore)
    sorted_d = sorted(FinalPathScore.items(), key=lambda x: x[1],reverse=True)
    BestPath=sorted_d[0][0]
    return (BestPath,FinalPathScore)
    #return (bestPath, mergedPathScores)
    #raise NotImplementedError




