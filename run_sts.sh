#!/bin/bash
model=${MODEL:-princeton-nlp/sup-simcse-roberta-large}
encoding=${ENCODER_TYPE:-bi_encoder}
lr=${LR:-0.00001}
wd=${WD:-0.1}
transform=${TRANSFORM:-False}
objective=${OBJECTIVE:-mse}
triencoder_head=${TRIENCODER_HEAD:-None}
seed=${SEED:-42}
output_dir=${OUTPUT_DIR:-output}
config=enc_${encoding}__lr_${lr}__wd_${wd}__trans_${transform}__obj_${objective}__tri_${triencoder_head}__s_${seed}
train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}

python run_sts.py \
  --output_dir "${output_dir}/${model//\//__}/${config}" \
  --model_name_or_path ${model} \
  --objective ${objective} \
  --encoding_type ${encoding} \
  --pooler_type cls \
  --freeze_encoder False \
  --transform ${transform} \
  --triencoder_head ${triencoder_head} \
  --max_seq_length 512 \
  --train_file ${train_file} \
  --validation_file ${eval_file} \
  --test_file ${test_file} \
  --condition_only False \
  --sentences_only False \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate ${lr} \
  --weight_decay ${wd} \
  --max_grad_norm 0.0 \
  --num_train_epochs 3 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --log_level info \
  --disable_tqdm True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --seed ${seed} \
  --data_seed ${seed} \
  --fp16 True \
  --log_time_interval 15