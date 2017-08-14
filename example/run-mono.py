import os
import sys
import argparse

# parse argv
parser = argparse.ArgumentParser(description = '...')
parser.add_argument('--concave_penalty', action='store', dest='concave_penalty2', type=bool, default=False)
parser.add_argument('--flsa', action='store', dest='flsa', type=int, default=0)
parser.add_argument('--batch_size', action='store', dest='batch_size', default=256, type=int)
parser.add_argument('--lr', action='store', dest='lr', default=4e-4, type=float)
parser.add_argument('--decay', action='store', dest='decay', default=0.2, type=float)
parser.add_argument('--l2', action='store', dest='l2', default=4e0, type=float)
parser.add_argument('--lconcave', action='store', dest='lconcave2', default=8e-1, type=float)
args=parser.parse_args()

task="train"
model_out="toy"
data_in="train"
val_data="validation"
nthreads=8
l1=1e-3
concave_penalty1=False
lconcave1=2e-1
feat_thresh=0
max_num_epochs=100
init_hrate=5e-2
epsilon1=1e-2
epsilon2=5e-2
epsilon=1e-4

#for lr in [4e-4]:
#    for l1 in [0]:
#        for l2 in [4e0]:
#            for lconcave1 in [2e-1]:
#                for lconcave2 in [8e-1]:
#                    for epsilon1 in [1e-2]:
#                        for epsilon2 in [5e-2]:
#                            for decay in [0.2]:

print '../build/hazard lr=%e decay=%e l1=%e l2=%e lconcave1=%e lconcave2=%e init_hrate=%e epsilon1=%e epsilon2=%e task=%s data_in=%s val_data=%s nthreads=%d batch_size=%d feat_thresh=%d max_num_epochs=%d concave_penalty1=%d concave_penalty2=%d flsa=%d epsilon=%e model_out=%s'%(args.lr,args.decay,l1,args.l2,lconcave1,args.lconcave2,init_hrate, epsilon1, epsilon2, task, data_in, val_data, nthreads, args.batch_size, feat_thresh, max_num_epochs, concave_penalty1, args.concave_penalty2, args.flsa, epsilon, model_out)
sys.stdout.flush()
os.system('../build/hazard lr=%e decay=%e l1=%e l2=%e lconcave1=%e lconcave2=%e init_hrate=%e epsilon1=%e epsilon2=%e task=%s data_in=%s val_data=%s nthreads=%d batch_size=%d feat_thresh=%d max_num_epochs=%d concave_penalty1=%d concave_penalty2=%d flsa=%d epsilon=%e model_out=%s'%(args.lr,args.decay,l1,args.l2,lconcave1,args.lconcave2,init_hrate, epsilon1, epsilon2, task, data_in, val_data, nthreads, args.batch_size, feat_thresh, max_num_epochs, concave_penalty1, args.concave_penalty2, args.flsa, epsilon, model_out))
sys.stdout.flush()


