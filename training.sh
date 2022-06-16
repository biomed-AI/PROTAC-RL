dataset_name=ZINC500
prior_step=300000
random=random
exp_id=1

pathsave=checkpoints/${dataset_name}/${random}
if [ ! -d "$pathsave"]; then
  mkdir $pathsaves
fi
mkdir $pathsave

CUDA_VISIBLE_DEVICES=3 python  train.py -data data/${dataset_name}/${random}/ \
                 -save_model checkpoints/${dataset_name}/${random}/SyntaLinker_prior -world_size 1 \
                 -valid_steps 10000 -seed 42 -gpu_ranks 0 -save_checkpoint_steps 50000 -keep_checkpoint 50 \
                 -train_steps ${prior_step}  -param_init 0  -param_init_glorot -max_generator_batches 96 \
                 -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                 -learning_rate 2 -label_smoothing 0.0 -report_every 1000 -log_file log/transformer_${dataset_name}_${random}_${exp_id}.txt\
                 -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                 -dropout 0.1 -position_encoding -share_embeddings \
                 -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                 -heads 8 -transformer_ff 2048

