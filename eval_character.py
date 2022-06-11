import argparse
import json

import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

CHARACTERS = [
    "BMO",
    "Rachel",
    "Burke",
    "Barney",
    "Spock",
    "Sheldon",
    "Dwight",
    "Michael",
    "BartSimpson",
    "MargeSimpson",
]
CHARACTER_TO_IDX = {c: i for i, c in enumerate(CHARACTERS)}


def transform_input(texts, tokenizer):
    result = tokenizer(texts, max_length=256, truncation=True, padding="max_length")
    result = {k: torch.LongTensor(v) for k, v in result.items()}
    # Optional when you run your model in GPU
    result = {k: v.to("cuda:0") for k, v in result.items()}
    return result


def run_model(texts, model, tokenizer):
    transformed = transform_input(texts, tokenizer)
    with torch.no_grad():
        logits = model(**transformed).logits
        logits = logits.cpu().numpy()
        probs = softmax(logits, axis=-1)
    return probs


def main(args):
    try:
        character_idx = CHARACTER_TO_IDX[args.character_name]
    except KeyError:
        raise ValueError(f"Unsupported character name: {args.character_name}")

    with open(args.input_path) as f:
        sentences = [json.loads(line.strip())["response"] for line in f]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, from_tf=False)
    # Optional when you run your model in GPU
    model = model.to("cuda:0")

    sum_probs = 0.
    num_instances = 0.

    for start_idx in range(0, len(sentences), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(sentences))
        batch = sentences[start_idx: end_idx]

        model_preds = run_model(batch, model, tokenizer)
        sum_probs += model_preds[:, character_idx].sum()
        num_instances += len(model_preds)

    avg_prob = sum_probs / num_instances
    print(f"Avg prob for predicting as character {args.character_name}: {avg_prob:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--character-name", type=str, choices=CHARACTERS)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    main(args)
