import random
import math
from math import log
random.seed()

#model setting
nData = 1000
nFeat = 10
ratioFeatActive = 0.1
ratioFeatInData = 0.4
T = 10.0
pieceWiseHr = [2e-1]
pieceWiseT = [4.0]
ratioTrain = 0.8
measure = T*0.05
rPosNeg = 0.8

#output data and model
ftrain = open("train1", "w")
fval = open("validation1", "w")
fmodel = open("model1", "w")

#calculate some numbers based on model setting
nTrain = int(nData * ratioTrain)
nVal = nData - nTrain
nFeatActive = int(nFeat * ratioFeatActive)
nFeatNonAct = nFeat - nFeatActive
nFeatInData = int(nFeat * ratioFeatInData) # each data contain Uniform(nFeatInData,2*nFeatIndata-1) number of feats
nPieceHr = len(pieceWiseHr)

# hazard rate for each feature, with len = nPiece
hr = {}
# hacked time for each data
bl = {}
# data set
data = []

nPos = 0
nNeg = 0
stat = {}


# sample hacked time given a data and golden model
def sampleHackedTime(i):
    Hr = {}
    rank = []
    d = data[i]
    target = 0
    for f in d:
        if f not in hr:
            continue
        else:
            target = f
            for i in range(nPieceHr):
                t = hr[f][i]
                if t not in Hr:
                    Hr[t] = pieceWiseHr[i]
	    break
    if len(Hr) == 0:
        return T
    rank = sorted(Hr)
#    prev = 0
#    for i in range(len(rank)):
#        Hr[rank[i]] += prev
#        prev = Hr[rank[i]]

    if target not in stat:
        stat[target] = [0]*(len(rank)+1)
    #sample
    i = 0
    while True:
        z = random.random()
        if z == 0.0:
            continue
        t = -log(z)/Hr[rank[i]]
        if (i+1>=len(rank) and t+rank[i]<=T) or (i+1<len(rank) and t+rank[i] <= rank[i+1]):
            stat[target][i] += 1
            return t+rank[i]
        else:
            i+=1
            if i >= len(rank):
                stat[target][i]+=1
                return T

# sample interval censored time given measure accurarcy
def sample_interval(hackt):
    lt = math.floor(hackt*10)/10
    rt = lt+0.1
#    sbegin = random.random() * measure
#    lt = max(0, sbegin + hackt - measure)
#    rt = min(T, lt + measure)
    if lt>=rt or lt<0 or rt>10:
        print 'error'
    return (rt,lt)

def save_data(start, nd, fw):
    global nPos, nNeg
    for i in range(start,nd):
        hackt = bl[i]
        if hackt == T: #right censored
            label = 0
            fw.write("%d\t%e\t%e"%(label, hackt, 0))
            nNeg += 1
        else:
            label = 1
            rt,lt = sample_interval(hackt)
            fw.write("%d\t%e\t%e"%(label, rt, lt))
            nPos += 1
        for f in data[i]:
            fw.write("\t%d"%f)
        fw.write("\n")
    fw.close()

def save_model(fw):
    for k in sorted(hr):
        fw.write("%d"%k)
        for t in range(nPieceHr):
            fw.write("\t%e:%e"%(hr[k][t],pieceWiseHr[t]))
        fw.write("\n")
    fw.close()

# generate hazard rate function for each active feature
def genHr():
    for i in range(nFeatActive):
        hr[i] = []
        for j in range(nPieceHr):
            hr[i].append(pieceWiseT[j])
#    	    while True:
#                tmp = random.random()*T
#                if tmp not in hr[i]:
#                    hr[i].append(tmp)
#                    break
        hr[i].sort()

# generate data
def genData():
    for i in range(nData):
        data.append([])
        data[i] = []
        avrgFeat = random.randint(nFeatInData, nFeatInData*2-1)
        if random.random() >= rPosNeg:
            data[i].append(random.randint(0,nFeatActive-1))
        for j in range(avrgFeat):
            while True:
                tmpf = random.randint(nFeatActive, nFeat-1)
                if tmpf not in data[i]:
                    data[i].append(tmpf)
                    break

# sample hacked time for data
def sample():
    for i in range(nData):
        bl[i] = sampleHackedTime(i)

def main():
    genHr()
    genData()
    sample()
    save_data(0,nTrain,ftrain)
    save_data(nTrain,nData,fval)
    save_model(fmodel)

def printStat():
    global nPos, nNeg
    print "#Pos = %d, #Neg = %d\n"%(nPos, nNeg)
    for f in stat:
        for i in range(len(stat[f])):
            print stat[f][i],
        print '\n'


if __name__ == "__main__":
    main()
    printStat()
