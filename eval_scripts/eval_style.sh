#!/bin/zsh

EVAL_INPUT_PATH=$1  # Path for input jsonl file containing pairs for evaluation
EXPECTED_STYLE_LABEL=$2
CLASSIFIER_MODEL_DIR=$3

python3 eval_style.py \
  --model-dir $CLASSIFIER_MODEL_DIR \
  --input-path $EVAL_INPUT_PATH\
  --expected-label $EXPECTED_STYLE_LABEL