
run_once(){
  model=$(basename $model_path)
  if [ "$lr_att_mode" = "none" ]; then
    if [ "$prepare_mlvu" = "1" ]; then
      name=prepare_mlvu_tmp
    else
      name=full
    fi
  elif [[ "$lr_att_mode" == infinigen* ]]; then
    name=b${budget}_p${lr_ratio}_${start_layer}_tg${token_group}
  elif [[ "$lr_att_mode" == shadowkv* ]]; then
    name=b${budget}
  else
    name=b${budget}_p${lr_ratio}_${start_layer}_tg${token_group}
  fi

  task_dir="${tasks%-*}"
  log_dir_=${log_dir}/${task_dir}/${model}_${seq_len}/${lr_att_mode}
  mkdir -p $log_dir_
  save_dir_=${save_dir}/${task_dir}/${model}_${seq_len}/${lr_att_mode}
  log_file=${log_dir_}/${name}.log
  save_name=${save_dir_}/${name}

  SKIP_MODEL_RUN=${SKIP_MODEL_RUN:-0}
  if [ "$SKIP_MODEL_RUN" = "1" ]; then
    echo "SKIP_MODEL_RUN=1, skipping model run"
    return 
  fi

  if [ "$lr_att_mode" = "none" ]; then
    if [ "$prepare_mlvu" = "1" ]; then
      python src/eval_gen.py --model_path ${model_path} \
        --tasks ${tasks} \
        --seq_len ${seq_len} \
        --eval_samples ${eval_samples} \
        --lr_att_mode ${lr_att_mode} \
        --save_video_emb \
        --save_name ${save_name} > $log_file
    else
      python src/eval_gen.py --model_path ${model_path} \
        --tasks ${tasks} \
        --seq_len ${seq_len} \
        --eval_samples ${eval_samples} \
        --lr_att_mode ${lr_att_mode} \
        --save_name ${save_name} > $log_file
    fi
  elif [[ "$lr_att_mode" == infinigen* ]]; then
    python src/eval_gen.py --model_path ${model_path} \
      --tasks ${tasks} \
      --seq_len ${seq_len} \
      --eval_samples ${eval_samples} \
      --skewing_matrix_path ${skewing_matrix_path} \
      --skewing_idx_path ${skewing_idx_path} \
      --lr_att_mode ${lr_att_mode} \
      --budget ${budget} \
      --save_name ${save_name} \
      --start_layer ${start_layer} \
      --token_group ${token_group} > $log_file
  elif [[ "$lr_att_mode" == shadowkv* ]]; then
    python src/eval_gen.py --model_path ${model_path} \
      --tasks ${tasks} \
      --seq_len ${seq_len} \
      --eval_samples ${eval_samples} \
      --lr_att_mode ${lr_att_mode} \
      --budget ${budget} \
      --save_name ${save_name} > $log_file
  else
    if [[ "$lr_att_mode" == loki* ]]; then
      lr_proj_path_=${loki_proj_path}
    else
      lr_proj_path_=${lr_proj_path}
    fi
    python src/eval_gen.py --model_path ${model_path} \
      --tasks ${tasks} \
      --seq_len ${seq_len} \
      --eval_samples ${eval_samples} \
      --lr_proj_path ${lr_proj_path_} \
      --lr_att_mode ${lr_att_mode} \
      --budget ${budget} \
      --save_name ${save_name} \
      --start_layer ${start_layer} \
      --token_group ${token_group} > $log_file
  fi
}


log_dir=./exps/logs
save_dir=./exps/results
mkdir -p $log_dir


MODEL_BASE_DIR=$PWD/MODELS

tasks=$1 
model_name=$2
lr_att_mode=$3
lr_ratio=$4
token_group=$5
budget=$6
start_layer=$7
eval_samples=${8:-10}
seq_len=${9:-32768}


echo "tasks="$tasks
echo "model_name="$model_name
echo "lr_att_mode="$lr_att_mode
echo "lr_ratio="$lr_ratio
echo "token_group="$token_group
echo "budget="$budget
echo "start_layer="$start_layer
echo "eval_samples="$eval_samples
echo "seq_len="$seq_len

source .venv/bin/activate

if [[ "$tasks" == prepare_mlvu ]]; then
  echo "prepare_mlvu=================================================="
  prepare_mlvu=1
  lr_att_mode=none
  tasks=mlvu
elif [[ "$tasks" == mlvu ]]; then
  echo "mlvu=================================================="
  PREPROCESS_DATASET=mlvu
  PREPROCESS_LEN=20
  prepare_mlvu=0
else
  prepare_mlvu=0
  PREPROCESS_DATASET=c4
  PREPROCESS_LEN=20
fi

lr_mode="${lr_att_mode##*_}"

lr_proj_wei_dir=./adapters/lowrank_proj_post_rope_${PREPROCESS_DATASET}_${PREPROCESS_LEN}
loki_proj_wei_dir=./adapters/loki_proj_post_rope_${PREPROCESS_DATASET}_${PREPROCESS_LEN}

skewing_matrix_dir=./adapters/infinigen_skew/skewing_matrix_${PREPROCESS_DATASET}_${PREPROCESS_LEN}
skewing_idx_dir=./adapters/infinigen_skew/skewing_idx_${PREPROCESS_DATASET}_${PREPROCESS_LEN}

lr_proj_path=${lr_proj_wei_dir}/${model_name}_${lr_mode}_${lr_ratio}
loki_proj_path=${loki_proj_wei_dir}/${model_name}_loki_${lr_ratio}
skewing_matrix_path=${skewing_matrix_dir}/${model_name}.pt
skewing_idx_path=${skewing_idx_dir}/${model_name}_${lr_ratio}
model_path=$MODEL_BASE_DIR/$model_name
run_once


if [[ "$tasks" == longbench* ]]; then
    python bench/longbench_gen_result.py \
    --results_dir ${save_dir_} \
    --prefix ${name}
elif [[ "$tasks" == needle ]]; then
    bash bench/Needle_test/eval.sh "${save_dir_}" > "${log_file}.needle" 2>&1 
elif [[ "$tasks" == ruler* ]]; then
    python bench/RULER/eval.py --results_dir ${save_dir_} --prefix ${name}
elif [[ "$tasks" == mlvu && "$prepare_mlvu" == 0 ]]; then
    bash bench/MLVU/eval.sh ${save_dir_} ${name}
fi

