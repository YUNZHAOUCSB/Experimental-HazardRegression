import os
import sys
import argparse

# parse argv
parser = argparse.ArgumentParser(description = '...')
parser.add_argument('--flsa', action='store', dest='flsa', type=int, default=0)
args=parser.parse_args()

task="train"
model_out="toy"
data_in="train_non"
val_data="validation_non"
nthreads=8
lr=1e-4
l1=1e-3
l2=1e-2
concave_penalty1=False
concave_penalty2=False
lconcave1=5e-3
lconcave2=5e-1
batch_size=256
feat_thresh=0
max_num_epochs=100
init_hrate=5e-2
epsilon1=1
epsilon2=1
decay=0
epsilon=1e-4

for lr in [2e-3]:
    for l1 in [0]:
        for l2 in [4e0]:
            for lconcave1 in [2e-1]:
                for lconcave2 in [5e0]:
                    for epsilon1 in [1e-2]:
                        for epsilon2 in [1]:
                            for decay in [0]:
                                print '../build/hazard lr=%e decay=%e l1=%e l2=%e lconcave1=%e lconcave2=%e init_hrate=%e epsilon1=%e epsilon2=%e task=%s data_in=%s val_data=%s nthreads=%d batch_size=%d feat_thresh=%d max_num_epochs=%d concave_penalty1=%d concave_penalty2=%d flsa=%d epsilon=%e model_out=%s'%(lr,decay,l1,l2,lconcave1,lconcave2,init_hrate, epsilon1, epsilon2, task, data_in, val_data, nthreads, batch_size, feat_thresh, max_num_epochs, concave_penalty1, concave_penalty2, args.flsa, epsilon, model_out)
                                sys.stdout.flush()
                                os.system('../build/hazard lr=%e decay=%e l1=%e l2=%e lconcave1=%e lconcave2=%e init_hrate=%e epsilon1=%e epsilon2=%e task=%s data_in=%s val_data=%s nthreads=%d batch_size=%d feat_thresh=%d max_num_epochs=%d concave_penalty1=%d concave_penalty2=%d flsa=%d epsilon=%e model_out=%s'%(lr,decay,l1,l2,lconcave1,lconcave2,init_hrate, epsilon1, epsilon2, task, data_in, val_data, nthreads, batch_size, feat_thresh, max_num_epochs, concave_penalty1, concave_penalty2, args.flsa, epsilon, model_out))
                                sys.stdout.flush()
