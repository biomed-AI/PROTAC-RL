#!/usr/bin/env bash
# this bash file used for reinforcement learning

dataset_name=PROTAC
# choose correct dictionary
random=random
# canonical/random for representation of SMILES
score_function=PK
# corresponding to scoring function in onmt/reinforce/scoring_functions.py
goal=20
# goal of scoring function if needed


agent_step=2000
# training steps of agents
ZINC_step=200000
PROTAC_step=3000
# choose prior

beamsize=50
# useless
train_nbest=50
# generate train_nbest SMILES in a batch for reinforcement learning
train_type=M
# M/B generation method: multinomial sampling/beam search

sigma=60
# coefficient for reinforcement learning, sometimes should be larger

exp_id=0
chooseGpu=0
lr=0.00001

model=Agent_on_${dataset_name}_zinc${ZINC_step}_protac${PROTAC_step}_${random}_${train_type}_${score_function}
case=dBET6
# corresponding to case folder name

pathsave=case/${case}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi

mkdir $pathsave
pathsave=checkpoints/${dataset_name}/Agent
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi

mkdir $pathsave
pathsave=checkpoints/${dataset_name}/Agent/${case}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
mkdir $pathsave

if [ "$train_type" == 'B' ];then
run_script=train_agent.py
else
run_script=train_agent_ms.py
fi
echo "training RL model"
echo ${run_script}

CUDA_VISIBLE_DEVICES=$chooseGpu python ${run_script} \
                    -model checkpoints/${dataset_name}/${random}/SyntaLinker_zinc${ZINC_step}_protac_step_${PROTAC_step}.pt \
                    -save_model checkpoints/${dataset_name}/Agent/${case}/Model_${model} \
                    -src case/${case}/src-test \
                    -tgt case/${case}/tgt-test \
                    -output case/${case}/${model}_batchsize${train_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 250 -beam_size $beamsize -verbose -n_best $train_nbest \
                    -gpu 0 -log_probs -log_file case/${case}/${model}_beamsize${beamsize}_batchsize${train_best}.txt \
                    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method fixed -warmup_steps 0  \
                    -rnn_size 256 -learning_rate $lr -label_smoothing 0.0 -report_every 20 \
                    -max_grad_norm 0 -save_checkpoint_steps 200 -train_steps $agent_step \
                    -scoring_function $score_function -score_function_num_processes 2 -sigma $sigma -goal $goal\
                    -gpu_ranks 0