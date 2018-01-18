#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/data-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/inception_v3_multiscale_lr_5loss
# Where the dataset is saved to.
DATASET_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501
# WHere the log is saved to
LOG_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/log
# Wher the tfrecord file is save to
OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_train
python train_multiscale.py \
--learning_rate=2e-4 \
--learning_rate_decay_type=exponential \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=100000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=360 \
--log_every_n_steps=100 \
--optimizer=adam \
--adam_beta1=0.5 \
--adam_beta2=0.999 \
--weight_decay=0.00004 \
--GPU_use=7 \
--pretrain_branch_0_path=/world/data-gpu-94/sysu-reid/checkpoints/inception_v3_299/model.ckpt-59286 \
--pretrain_branch_1_path=/world/data-gpu-94/sysu-reid/checkpoints/inception_v3_225/model.ckpt-58626
