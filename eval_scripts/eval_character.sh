#!/bin/zsh

EVAL_INPUT_PATH=$1  # Path for input jsonl file containing pairs for evaluation
CHARACTER_NAME=$2
CLASSIFIER_MODEL_DIR=$3

python3 eval_character.py \
  --model-dir $CLASSIFIER_MODEL_DIR \
  --input-path $EVAL_INPUT_PATH \
  --character-name $CHARACTER_NAME
