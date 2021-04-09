#!/bin/bash

proxychains4 python train.py \
/media/drs/extra/Datasets/MVI \
--gpu 0 \
--lr 0.0001 \
--epochs 50 \
-b 8 \

