import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from termcolor import colored

from methods.inference import build_faiss_index
from methods.inference import get_candidate_embeddings
from methods.inference import get_pseudo_context_embeddings
from methods.inference import get_reference_exemplars_using_answer
from methods.inference import load_agent
from methods.inference import read_exemplars
from methods.inference import select_response


def main(args, agent):
    whole_exemplars = []
    for style in args.styles:
        styled_exemplars = []
        styled_exemplars.extend(read_exemplars(args.all_styles_path, 8, style))
        whole_exemplars.append(styled_exemplars)
    test_samples = pd.read_csv(args.evaluate_set, delimiter="\t")

    # replace exemplar questions with PSEUDO CONTEXT.
    with open(args.pseudo_context_path, "r") as f:
        pseudo_contexts = f.read().split("\n")[:-1]
        print(f"Reading pseudo contexts: {len(pseudo_contexts)} contexts.")
    emb_path = Path(args.pseudo_context_path).with_suffix(".npy")
    if emb_path.exists():
        print("Embedding already exists.")
        pseudo_context_embeddings = torch.Tensor(np.load(str(emb_path), allow_pickle=False)).half().cuda()
    else:
        print("Saving embeddings.")
        pseudo_context_embeddings = get_pseudo_context_embeddings(agent, pseudo_contexts)
        np.save(str(emb_path), pseudo_context_embeddings.cpu().numpy())
    faiss_index = build_faiss_index(pseudo_context_embeddings.cpu().numpy().astype(np.float32))
    batch_size = 128
    whole_candidate_embeddings = []
    for exemplars in whole_exemplars:
        for start_idx in range(0, len(exemplars), batch_size):
            end_idx = min(len(exemplars), start_idx + batch_size)
            batch = [exemplar["response"] for exemplar in exemplars[start_idx: end_idx]]
            response_emb = get_candidate_embeddings(agent, batch)
            faiss_score, cand_indices = faiss_index.search(response_emb.cpu().numpy().astype(np.float32), 1)
            for i in range(len(batch)):
                if cand_indices[i, 0] == -1:
                    exemplars[start_idx + i]["context"] = "Hello."
                else:
                    exemplars[start_idx + i]["context"] = pseudo_contexts[cand_indices[i, 0]]
        whole_candidate_embeddings.append(get_candidate_embeddings(agent, [exemplar["response"] for exemplar in exemplars]))

    results = []
    for _, row in list(test_samples.iterrows()):
        context = [row.Query]
        curr_exemplars = []
        for num_exemplars, styled_exemplars, candidate_embeddings in zip(args.max_num_exemplars, whole_exemplars, whole_candidate_embeddings):
            curr_exemplars.extend(get_reference_exemplars_using_answer(
                agent, context, styled_exemplars, candidate_embeddings, int(num_exemplars)))
        curr_exemplars = list(sorted(curr_exemplars, key=lambda x: x["score"]))
        if args.print_docs:
            for exp in curr_exemplars:
                print(colored(exp, "cyan"))

        prefix_context = []
        for exemplar in curr_exemplars:
            prefix_context.extend([f"User: {exemplar['context']}", f"{args.character_name}: {exemplar['response']}"])

        prefix_context = f"This is conversation between User and {args.character_name}.\n" + "\n".join(prefix_context) + "\n"
        context_str = "\n".join([f"User: {utterance}" if i % 2 == 0 else f"{args.character_name}: {utterance}"
                                 for i, utterance in enumerate(context)])

        server_response = requests.post(
            args.megatron_endpoint,
            data=json.dumps({
                "context": prefix_context + context_str + f"\n{args.character_name}:",
            })
        )
        assert server_response.status_code == 200

        candidates = json.loads(server_response.text)
        if args.print_docs:
            for cand in candidates:
                print(colored(cand, "yellow"))
        response_str = select_response(agent, context, candidates, args.response_selection_strategy)
        print(f"Input: {context}")
        print(colored(f"{args.character_name}:", "blue", attrs=["bold"]),
              colored(response_str, "white", attrs=["bold"]))
        results.append((context, response_str))

    if args.save_results_path:
        Path(args.save_results_path).mkdir(exist_ok=True, parents=True)
        with open(str(Path(args.save_results_path) / f"ablation-noaug-{args.styles[0]}{args.max_num_exemplars[0]}.jsonl"), "w") as f:
            for result in results:
                json.dump({"context": result[0], "response": result[1]}, f)
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--all-styles-path", type=str)
    parser.add_argument("--megatron-endpoint", type=str, required=True)
    parser.add_argument("--max-num-exemplars", nargs='+', default=[])
    parser.add_argument("--character-name", type=str, default="Bot")
    parser.add_argument("--response-selection-strategy", type=str,
                        choices=["random", "top1", "ranker"], default="ranker")
    parser.add_argument("--print-docs", action="store_true")
    parser.add_argument("--pseudo-context-path", type=str, default="./resources/predefined_texts.txt")
    parser.add_argument("--evaluate-set", type=str)
    parser.add_argument("--styles", nargs='+', default=[], required=True)
    parser.add_argument("--save-results-path", default=None, type=str)

    args = parser.parse_args()
    agent = load_agent(args.model_file)
    random.seed(777)
    print(colored(f"\n\n[Current style] {args.styles}\n\n", "blue", attrs=["bold"]))
    main(args, agent)
