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
# CUDA_VISIBLE_DEVICES=3,4 python train_ade.py -c configs/config_ade.json \
# -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

# CUDA_VISIBLE_DEVICES=1,2,3,4 python train_ade.py -c configs/config_ade.json \
# -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=1,2,3,4 python eval_ade.py -d 0 --test -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_1/checkpoint-epoch${EPOCH}.pth