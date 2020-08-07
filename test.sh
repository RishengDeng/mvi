#!/bin/bash

python test.py \
/media/drs/extra/Datasets/mvi_data/art_npy \
--gpu 0 \
--lr 0.0001 \
-b 1 \
--resume '/media/drs/extra/Learn/code/mvi/ckpts/0806/dl_drn22_testbest.pth.tar' \