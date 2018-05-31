#!/bin/sh
#to run on wheel data
module load python/2.7.6
module load tensorflow/11-python2.7

python getHeight.py > S2_fl_1130_630amm.txt
#python getHeight.py > S2_fl__forward_1130_630am.txt
#python getHeight.py > S2_L2_1129_5pm.txt
#python getHeight.py > S1_L2_1129_5pm.txt
