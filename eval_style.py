import argparse
import json

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


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
    preds = np.argmax(probs, axis=-1)
    return preds


def main(args):
    LABEL_STR_TO_INT = {
        "modern": 0,
        "shakespearen": 1,
        "negative": 0,
        "positive": 1,
        "anger": 0,
        "joy": 1,
    }
    expected_label = LABEL_STR_TO_INT[args.expected_label]

    with open(args.input_path) as f:
        sentences = [json.loads(line.strip())["response"] for line in f]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, from_tf=False)
    # Optional when you run your model in GPU
    model = model.to("cuda:0")

    num_right = 0

    for start_idx in range(0, len(sentences), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(sentences))
        batch = sentences[start_idx: end_idx]

        model_preds = run_model(batch, model, tokenizer)
        num_right += (model_preds == expected_label).sum()

    accuracy = num_right * 100. / len(sentences)
    print(f"Accuracy: {accuracy:.6f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--expected-label", type=str, choices=["modern",
                                                               "shakespearen",
                                                               "negative",
                                                               "positive",
                                                               "anger",
                                                               "joy"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    main(args)
