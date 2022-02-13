dataset_name=PROTAC
ZINC_step=300000
# step of pre-training model for initial fine-tuning model
protac_step=7000
# fine-tuning step
random=random
model=SyntaLinker_prior_step
pathsave=checkpoints/${dataset_name}/${random}/
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
mkdir $pathsave
CUDA_VISIBLE_DEVICES=3 python  train.py -data data/${dataset_name}/${random}/ \
                 -train_from checkpoints/ZINC500/${random}/${model}_${ZINC_step}.pt \
                 -save_model checkpoints/${dataset_name}/${random}/SyntaLinker_zinc${ZINC_step}_protac -world_size 1 \
		             -valid_steps 1000 -seed 42 -gpu_ranks 0 -save_checkpoint_steps 1000 -keep_checkpoint 50 \
                 -train_steps ${protac_step}  -param_init 0  -param_init_glorot -max_generator_batches 96 \
                 -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                 -learning_rate 2 -label_smoothing 0.0 -report_every 1000 -log_file log/transformer_${dataset_name}_${random}_protac${protac_step}_zinc${ZINC_step}.txt\
                 -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                 -dropout 0.1 -position_encoding -share_embeddings \
                 -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                 -heads 8 -transformer_ff 2048