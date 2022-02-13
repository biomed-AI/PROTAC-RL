#!/usr/bin/env bash
# this file used for generating SMILES with prior using multinomial sampling in one case

dataset_name=PROTAC
random=random

ZINC_step=200000
PROTAC_step=3000

beamsize=32
test_nbest=250

case=dBET6
model=SyntaLinker_zinc${ZINC_step}_protac_step_${PROTAC_step}

chooseGpu=2


echo "beam search using agent model"
CUDA_VISIBLE_DEVICES=$chooseGpu python translate_ms.py -model checkpoints/${dataset_name}/${random}/${model}.pt \
                    -src case/${case}/src-test \
                    -tgt case/${case}/tgt-test \
                    -output case/${case}/predictions_prior_ms_${test_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $test_nbest \
                    -gpu 0 -log_probs -log_file ''



