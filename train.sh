#!/bin/bash

proxychains4 python train.py \
/media/drs/extra/Datasets/mvi_data/patch_npy \
--gpu 0 \
--lr 0.0001 \
--epochs 30 \
-b 6 \

