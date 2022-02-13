#!/usr/bin/env bash
# this file used for generating SMILES with model having been trained with reinforcement learning using beam search
dataset_name=PROTAC
random=canonical

agent_step=1000
ZINC_step=200000
PROTAC_step=3000

beamsize=50
test_nbest=500

train_type=M
score_function=PK
case=dBET6

model=Model_Agent_on_${dataset_name}_zinc${ZINC_step}_protac${PROTAC_step}_${random}_${train_type}_${score_function}

chooseGpu=1

CUDA_VISIBLE_DEVICES=$chooseGpu python translate.py -model checkpoints/${dataset_name}/Agent/${case}/${model}_step_${agent_step}.pt \
                    -src case/${case}/src-test \
                    -tgt case/${case}/tgt-test \
                    -output case/${case}/predictions_agent_beam50_nbest500.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size ${beamsize} -verbose -n_best ${test_nbest} \
                    -gpu 0 -log_probs -log_file ''
