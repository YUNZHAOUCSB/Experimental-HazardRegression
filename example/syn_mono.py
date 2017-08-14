import random
import math
from math import log
random.seed()

#model setting
nData = 1000
nFeat = 10
ratioFeatActive = 0.1
ratioFeatInData = 0.3
T = 10.0
nPiece=3
hrMean = 2e-1
hrVar = 0.1
ratioTrain = 0.8
rPosNeg = 0.2
epsilon=1e-3

#output data and model
ftrain = open("train", "w")
fval = open("validation", "w")
fmodel = open("model", "w")

#init setting
nTrain = int(nData * ratioTrain)
nVal = nData - nTrain
nFeatActive = int(nFeat * ratioFeatActive)
nFeatNonAct = nFeat - nFeatActive
nFeatInData = int(nFeat * ratioFeatInData) # each data contain Uniform(nFeatInData,2*nFeatIndata-1) number of feats
pieceWiseHr = []
pieceWiseT = []
interv=float(T/nPiece)

# hacked time for each data
bl = {}
# data set
data = []

nPos = 0
nNeg = 0
stat = {}


# sample hacked time given a data and golden model
def sampleHackedTime(id):
    d = data[id]
    Hr = {}
    rank = []
    for f in d:
        if f >= nFeatActive:
            continue
        else:
            if len(Hr) == 0:
                for i in range(nPiece):
                    Hr[pieceWiseT[f][i]] = pieceWiseHr[f][i]
            else:
                pp = 0
                for i in range(nPiece):
                    t = pieceWiseT[f][i]
                    r = pieceWiseHr[f][i]
                    if t in Hr:
                        for k in sorted(Hr):
                            if k>=t:
                                Hr[k]+=r-pp
                    else:
                        prev = t
                        for k in sorted(Hr):
                            if k > t:
                                break
                            else:
                                prev = k
                        if prev in Hr:
                            Hr[t] = Hr[prev]
                        else:
                            Hr[t] = 0
                        for k in sorted(Hr):
                            if k>=t:
                                Hr[k]+=r-pp
                    pp = r
    if len(Hr) == 0:
        return T
    rank = sorted(Hr)

    #debug
    if len(Hr) != 0:
        print '%d'%id
        for t in sorted(Hr):
            print '\t%e:%e'%(t,Hr[t])
    print '\n\n\n'

    #sample
    i = 0
    while True:
        z = random.random()
        if z == 0.0:
            continue
        t = -log(z)/Hr[rank[i]]
        if (i+1>=len(rank) and t+rank[i]<=T) or (i+1<len(rank) and t+rank[i] <= rank[i+1]):
            return t+rank[i]
        else:
            i+=1
            if i >= len(rank):
                return T

# sample interval censored time given measure accurarcy
def sample_interval(hackt):
    lt = math.floor(hackt*10)/10
    rt = lt+0.1
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
    for k in range(nFeatActive):
        fw.write("%d"%k)
        for t in range(nPiece):
            fw.write("\t%e:%e"%(pieceWiseT[k][t],pieceWiseHr[k][t]))
        fw.write("\n")
    fw.close()

# generate hazard rate function for each active feature
def genHr():
    for i in range(nFeatActive):
        pieceWiseT.append([])
        for j in range(nPiece):
            l=j*interv
            pieceWiseT[i].append(l+random.random()*interv)
    for i in range(nFeatActive):
        pieceWiseHr.append([])
        for j in range(nPiece):
            pieceWiseHr[i].append(max(random.gauss(hrMean,hrVar),epsilon))
        pieceWiseHr[i].sort()

# generate data
def genData():
    for i in range(nData):
        data.append([])
        data[i] = []
	if rPosNeg > random.random():
	    np = random.randint(1,nFeatActive)
	    for k in range(np):
		while True:
		    tmpf = random.randint(0,nFeatActive-1)
		    if tmpf not in data[i]:
			data[i].append(tmpf)
			break
        avrgFeat = random.randint(nFeatInData, nFeatInData*2-1)
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


if __name__ == "__main__":
    main()
    printStat()
