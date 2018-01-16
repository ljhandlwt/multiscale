#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/data-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/inception_v3_299
# Where the dataset is saved to.
DATASET_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501
# WHere the log is saved to
LOG_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/log
# Wher the tfrecord file is save to
OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_train
python train_inception_single.py \
--learning_rate=2e-4 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=20000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=300 \
--log_every_n_steps=100 \
--optimizer=adam \
--adam_beta1=0.5 \
--adam_beta2=0.999 \
--weight_decay=0.00004 \
--scale_size=299 \
--GPU_use=6

python train_inception_single.py \
--learning_rate=5e-5 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=40000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=300 \
--log_every_n_steps=100 \
--optimizer=adam \
--adam_beta1=0.5 \
--adam_beta2=0.999 \
--weight_decay=0.00004 \
--scale_size=299 \
--GPU_use=6

python train_inception_single.py \
--learning_rate=1e-5 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=60000 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=300 \
--log_every_n_steps=100 \
--optimizer=adam \
--adam_beta1=0.5 \
--adam_beta2=0.999 \
--weight_decay=0.00004 \
--scale_size=299 \
--GPU_use=6
