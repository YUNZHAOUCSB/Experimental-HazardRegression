import random
import math
random.seed()
nData = 1000
nFeat = 1000
ratioFeatActive = 0.3
ratioFeatInData = 0.01
T = 10
lcensor = 1
nPieceHr = 3
rr = [6e-3, 1e-2, 1.2e-2, 1.2e-2]
ftrain = open("train", "w")
fval = open("validation", "w")
fm = open("model", "w")
ratioTrain = 0.8

nTrain = int(nData * ratioTrain)
nVal = nData - nTrain
nFeatActive = int(nFeat * ratioFeatActive)
nFeatNonAct = nFeat - nFeatActive
# \NN (nFeatInData, var)
nFeatInData = int(nFeat * ratioFeatInData)

# generate hr function for each active feature
hr = {}
for i in range(nFeatActive):
    hr[i] = []
    for j in range(nPieceHr):
	while True:
	    tmp = random.randint(lcensor,T-1)
	    if tmp not in hr[i]:
        	hr[i].append(tmp)
		break
    hr[i].append(T)
    hr[i].sort()

# generate data
data = []
for i in range(nData):
    data.append([])
    data[i] = []
    avrgFeat = random.randint(nFeatInData, nFeatInData*2-1)
    for j in range(avrgFeat):
        data[i].append(random.randint(0, nFeat-1))

# sample hacked time
def sampleHackedTime(i):
    Hr = [0.0] * T
    densities = [0.0] * T
    d = data[i]
    go_ahead = 0
    for f in d:
        k = 0
        if f not in hr:
            continue
        else:
            go_ahead = 1
            pivot = hr[f][0]
            hrate = 0.0
            for t in range(T):
                if t < pivot:
                    Hr[t] += hrate;
                else:
                    hrate = rr[k]
                    Hr[t] += hrate
                    k = min(k+1, nPieceHr)
                    pivot = hr[f][k]
    if not go_ahead:
        return T-1
    prev = 0.0
    for t in range(T):
        densities[t] = Hr[t] * math.exp(-prev);
        prev += Hr[t]
    # sample
    prev = 0.0
    for t in range(T):
        densities[t] = densities[t] + prev
        prev = densities[t]
    p = random.random()
    p = p * densities[T-1]
    for t in range(T):
        if densities[t] > p:
            return t

# generate blacklist
bl = {}
for i in range(nData):
    bl[i] = sampleHackedTime(i)

def save_data(start, nd, fw):
    for i in range(start,nd):
        rt = bl[i]
        if rt == T-1:
            lt = 0
            label = 0
            fw.write("%d\t%d\t%d"%(label, rt, lt))
        else:
            lt = rt - lcensor
            label = 1
            fw.write("%d\t%d\t%d"%(label, rt, lt))
        for f in data[i]:
            fw.write("\t%d"%f)
        fw.write("\n")
    fw.close()

def save_model():
    for k in sorted(hr):
        fm.write("%d"%k)
        for t in range(nPieceHr):
            fm.write("\t%d:%e"%(hr[k][t],rr[t]))
        fm.write("\n")
    fm.close()

save_data(0,nTrain,ftrain)
save_data(nTrain,nData,fval)
save_model()

