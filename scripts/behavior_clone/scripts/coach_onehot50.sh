# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python train_coach.py \
  --train_dataset ../../data/dataset/train.json \
  --val_dataset ../../data/dataset/val.json \
  --inst_dict_path ../../data/dataset/dict.pt \
  --emb_field_dim 32 \
  --prev_cmd_dim 64 \
  --num_conv_layers 3 \
  --num_post_layers 1 \
  --conv_hid_dim 256 \
  --army_out_dim 128 \
  --other_out_dim 128 \
  --money_hid_layer 1 \
  --conv_dropout 0.0 \
  --word_emb_dim 128 \
  --word_emb_dropout 0.25 \
  --inst_hid_dim 256 \
  --count_hid_dim 256 \
  --count_hid_layers 2 \
  --glob_dropout 0.5 \
  --coach_type onehot \
  --model_folder saved_models/coach_onehot50 \
  --batch_size 100 \
  --gpu 0 \
  --grad_clip 0.5 \
  --lr 0.002 \
  --optim adamax \
  --epochs 50 \
  --num_pos_inst 50 \
  --num_neg_inst 500 \
  --pos_dim 16 \
  --prev_cmd_rnn 1 \
  --seed 3 \
