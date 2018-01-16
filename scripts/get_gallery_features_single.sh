#!/bin/bash
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/world/date-gpu-94/sysu-reid/checkpoints
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/world/data-gpu-94/sysu-reid/checkpoints/inception_v3_299
# Where the dataset is saved to.
DATASET_DIR=/home/yuanziyi/Market-1501
# WHere the log is saved to
LOG_DIR=/home/yuanziyi/log
# Wher the tfrecord file is save to
OUTPUT_DIR=/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_test
python get_gallery_features_single.py \
--learning_rate=2e-3 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=6 \
--log_every_n_steps=5 \
--optimizer=sgd \
--weight_decay=0.00004 \
--ckpt_num=54964

python get_gallery_features_single.py \
--learning_rate=2e-3 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=6 \
--log_every_n_steps=5 \
--optimizer=sgd \
--weight_decay=0.00004 \
--ckpt_num=57181

python get_gallery_features_single.py \
--learning_rate=2e-3 \
--learning_rate_decay_type=fixed \
--dataset_name=Market_1501 \
--dataset_split_name=train \
--dataset_dir=${OUTPUT_DIR} \
--model_name=inception_v3 \
--batch_size=8 \
--max_number_of_steps=10001 \
--checkpoint_dir=${TRAIN_DIR} \
--pretrain_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
--log_dir=${LOG_DIR} \
--save_model_summary_secs=6 \
--log_every_n_steps=5 \
--optimizer=sgd \
--weight_decay=0.00004 \
--ckpt_num=59286
