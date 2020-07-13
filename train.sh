#!/bin/bash

proxychains4 python train.py \
/media/drs/extra/Datasets/mvi_data/art_npy_10 \
--gpu 0 \
--lr 0.0001 \
--epochs 50 \
-b 4 \

