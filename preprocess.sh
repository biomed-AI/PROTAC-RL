#!/usr/bin/env bash


dataset_name=PROTAC
random=random

python preprocess.py -train_src data/${dataset_name}/${random}/src-train \
                     -train_tgt data/${dataset_name}/${random}/tgt-train \
                     -valid_src data/${dataset_name}/${random}/src-val \
                     -valid_tgt data/${dataset_name}/${random}/tgt-val \
                     -save_data data/${dataset_name}/${random}/ \
                     -src_seq_length 3000 -tgt_seq_length 3000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
