

data_dir=./bench/MLVU/MVLU_DATA/MLVU
output_dir=$1
prefix=$2


python ./bench/MLVU/evaluation/generation_evaluation/evaluate.py \
    --data_path $data_dir \
    --task sub_scene \
    --pred_path $output_dir/${prefix}_subplot_all.json \
    --output_dir $output_dir/${prefix}_subplot_all_eval \
    --output_json $output_dir/${prefix}_subplot_all_eval.json \
    --num_tasks 8 > $output_dir/${prefix}_eval_ssc.log

python ./bench/MLVU/evaluation/generation_evaluation/evaluate.py \
    --data_path $data_dir \
    --task summary \
    --pred_path $output_dir/${prefix}_summary_all.json \
    --output_dir $output_dir/${prefix}_summary_all_eval \
    --output_json $output_dir/${prefix}_summary_all_eval.json \
    --num_tasks 8 > $output_dir/${prefix}_eval_summary.log

python ./bench/MLVU/evaluation/generation_evaluation/calculate.py \
    --path $output_dir/${prefix}_subplot_all_eval > $output_dir/${prefix}_subplot_all_eval.log

python ./bench/MLVU/evaluation/generation_evaluation/calculate_sum.py \
    --path $output_dir/${prefix}_summary_all_eval > $output_dir/${prefix}_summary_all_eval.log

