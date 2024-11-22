#!/bin/bash

GPU=1,2,3,4
BS=8  # Total 24
SAVEDIR='ade_addpseudo'

TASKSETTING='overlap'
TASKNAME='100-50'
EPOCH=100
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=300 # 300 for Ours-M

NAME='DKD'

CUDA_VISIBLE_DEVICES=1,2,3,4 python eval_ade.py -d 0 --test -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_1/checkpoint-epoch${EPOCH}.pth
