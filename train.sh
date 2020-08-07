#!/bin/bash

proxychains4 python train.py \
/media/drs/extra/Datasets/mvi_data/art_bbox_npy \
--gpu 0 \
--lr 0.0001 \
--epochs 55 \
-b 8 \

