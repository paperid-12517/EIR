#!/bin/bash

PORT='tcp://127.0.0.1:12345'
GPU=0,1,6,7
BS=8  # Total 24
SAVEDIR='ade_addpseudo'

TASKSETTING='overlap'
TASKNAME='50-50'
EPOCH=100
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=300 # 300 for Ours-M

NAME='DKD'
CUDA_VISIBLE_DEVICES=0,1,6,7 python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}
