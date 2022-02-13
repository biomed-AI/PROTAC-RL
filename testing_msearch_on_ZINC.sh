#!/usr/bin/env bash
# this file used for generating SMILES with model only trained on ZINC using multinomial sampling

dataset_name=ZINC500
model=SyntaLinker_prior_step
step=200000
random=random

CUDA_VISIBLE_DEVICES=0 python translate_ms.py -model checkpoints/${dataset_name}/${random}/${model}_${step}.pt \
                    -src data/${dataset_name}/${random}/src-test \
                    -tgt data/${dataset_name}/${random}/tgt-test \
                    -output checkpoints/results/predictions_${dataset_name}_${random}_${model}_${step}.txt \
                    -batch_size 64 -replace_unk -max_length 200 -verbose -n_best 10 \
                    -gpu 0 -log_probs -log_file log/translate_${dataset_name}_${random}_${model}_${step}_log.txt