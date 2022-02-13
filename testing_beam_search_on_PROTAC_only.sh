#!/usr/bin/env bash
# this file used for generating SMILES with model only trained on PROTAC using beam search

dataset_name=PROTAC
model=SyntaLinker_prior_step
step=50000
random=random

CUDA_VISIBLE_DEVICES=1 python translate.py -model checkpoints/${dataset_name}/${random}/${model}_${step}.pt \
                    -src data/${dataset_name}/${random}/${NL}/src-test \
                    -output checkpoints/results/predictions_PROTACONLY_${model}_${step}_on_${dataset_name}_beam50_nbest10.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size 20 -verbose -n_best 10 \
                    -gpu 0 -log_probs -log_file log/translate_logs_PROTACONLY_${model}_${step}_on_${dataset_name}.txt
