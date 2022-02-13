#!/usr/bin/env bash
# this file used for generating SMILES with prior in one case

dataset_name=PROTAC
random=canonical

ZINC_step=200000
PROTAC_step=3000

beamsize=50
test_nbest=500

case=dBET6
model=SyntaLinker_zinc${ZINC_step}_protac_step_${PROTAC_step}


CUDA_VISIBLE_DEVICES=2 python translate.py -model checkpoints/${dataset_name}/${random}/${model}.pt \
                    -src case/${case}/src-test \
                    -tgt case/${case}/tgt-test \
                    -output case/${case}/predictions_beam${beamsize}_nbest${test_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size ${beamsize} -verbose -n_best ${test_nbest} \
                    -gpu 0 -log_probs -log_file ''
