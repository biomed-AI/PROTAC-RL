#!/usr/bin/env bash
# this file used for generating SMILES with model fine-tuned on PROTAC using multinomial sampling


dataset_name=PROTAC
model=SyntaLinker
zinc_step=200000
protac_step=3000
random=random

CUDA_VISIBLE_DEVICES=0 python translate_ms.py -model checkpoints/${dataset_name}/${random}/${model}_zinc${zinc_step}_protac_step_${protac_step}.pt \
                    -src data/${dataset_name}/${random}/src-test \
                    -output checkpoints/results/predictions_zinc${zinc_step}_protac${protac_step}_${random}_beam50_nbest10.txt \
                    -batch_size 64 -replace_unk -max_length 200 -beam_size 20 -verbose -n_best 10 \
                    -gpu 0 -log_probs -log_file log/translate_logs_zinc${zinc_step}_protac${protac_step}_${random}.txt
