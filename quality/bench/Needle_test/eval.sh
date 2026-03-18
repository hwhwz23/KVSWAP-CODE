
EVAL_DIR=$1
python bench/Needle_test/eval.py --jsonl_dir $EVAL_DIR
python bench/Needle_test/vis.py --result_dir $EVAL_DIR

