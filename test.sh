#!/bin/bash

python test.py \
/media/drs/extra/Datasets/MVI/EV1_process/ \
--gpu 0 \
--lr 0.0001 \
-b 1 \
--resume '/media/drs/extra/Learn/code/mvi/ckpts/0220/' \