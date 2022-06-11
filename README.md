# Meet Your Favorite Character: Open-domain Chatbot Mimicking Fictional Characters with only a Few Utterances (NAACL 2022)

- [Paper link](https://arxiv.org/pdf/2204.10825.pdf)

## Env Setup
```
conda create -n [env_name] python=3.8
conda activate [env_name]

conda install cudatoolkit=11.0 pytorch=1.7.1
pip install git+https://github.com/huggingface/transformers
pip install datasets pandas pyarrow sklearn
pip install -r requirements.txt
```

## Run inference
- We assume you run your language model and launch server on `http://${1}/generate`.
- Below is an example for `Dynamic Match`.

```
python3 methods/inference_dynamic_match.py \
  --model-file $retriever_model_path \
  --megatron-endpoint http://${1}/generate \
  --character-name $character_name \
  --response-selection-strategy top1 \
  --max-num-exemplars 8 \
  --evaluate-set resources/dailydialog_test_utterances.tsv \
  --all-styles-path resources/all_styles.tsv \
  --save-results-path results \
  --styles $character
```

## Run Character StyleProb Evaluation
```
srun --gres=gpu:1 eval_scripts/eval_character.sh [jsonl_input_file_path] [character_name] [classfier_model_path]
```

## Run Other Styles StyleProb Evaluation
Styles including `positive`, `negative`, `Modern`, `Shakespearean`, `joy`, `anger`.
```
srun --gres=gpu:1 eval_scripts/eval_style.sh [jsonl_input_file_path] [expected_label] [classfier_model_path]
```

### Note: Example of Jsonl file
```
{"context": ["that's awesome! Do you spend a lot of time there?", "i do! it's a lot of fun but it can be tiring sometimes", "I can imagine. what kind of restaurant do they own?"], "response": "The restaurant the restaurant"}
{"context": ["I got some great news today! My husband got a better paying job offer!", "Holy cow that's awesome!!!  What are you going to do with all that extra moneys??", "Not sure yet, but itll help us life more comforatbly! We move to his hometown in November when he gets out of Army!"], "response": "You must be so thrilled. There are so many lonely life out there. He must be thrilled."}
...
```

## Run MaUdE
- Clone [MaUdE Repo](https://github.com/facebookresearch/online_dialog_eval) and setup environment
- Run following script

```
cat maude_inference.sh

>>>

#!/bin/zsh
MODEL_SAVE_DIR=full_runs/
DATA_NAME=convai2
DATA_LOC=$1
FINE_TUNE_MODEL=convai2_data/distilbert_lm
TRAIN_MODE=nce

VERSION=20488119
MODEL_ID=na_all


for DATA_LOC in "$@"
do
  python3 codes/inference.py \
    --id $MODEL_ID \
    --model_save_dir $MODEL_SAVE_DIR \
    --model_version $VERSION \
    --train_mode nce \
    --corrupt_pre $DATA_LOC \
    --test_suffix true_response \
    --test_column response
done
```
```
srun --gres=gpu:1 maude_inference.sh [jsonl_path]
```

## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publications.

```
@article{han2022meet,
  title={Meet Your Favorite Character: Open-domain Chatbot Mimicking Fictional Characters with only a Few Utterances},
  author={Han, Seungju and Kim, Beomsu and Yoo, Jin Yong and Seo, Seokjun and Kim, Sangbum and Erdenee, Enkhbayar and Chang, Buru},
  journal={arXiv preprint arXiv:2204.10825},
  year={2022}
}
```
