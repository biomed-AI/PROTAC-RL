#!/usr/bin/env bash
# this file used for generating SMILES with model only trained on ZINC using beam search

dataset_name=ZINC500
model=SyntaLinker_prior_step
step=200000
# training steps of model

random=random
# canonical/random for representation of SMILES

CUDA_VISIBLE_DEVICES=0 python translate.py -model checkpoints/${dataset_name}/${random}/${model}_${step}.pt \
                    -src data/${dataset_name}/${random}/src-test \
                    -output checkpoints/predictions_${model}_${step}_on_${random}_${dataset_name}_beam50_nbest10.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size 20 -verbose -n_best 10 \
                    -gpu 0 -log_probs -log_file log/translate_logs_${model}_${step}.txt