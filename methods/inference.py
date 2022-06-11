import os
import random
import sys
from typing import List

import faiss
import numpy as np
import pandas as pd
import torch
from parlai.agents.transformer.transformer import TransformerRankerAgent
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.utils import logging
from tqdm import tqdm


def load_agent(model_file: str):
    parser = ParlaiParser(False, False)
    TransformerRankerAgent.add_cmdline_args(parser)
    parser.set_params(
        model="transformer/biencoder",
        model_file=model_file,
        print_scores=False,
        data_parallel=False,
        no_cuda=False,
        gpu=0,
    )
    safety_opt = parser.parse_args([])
    agent = create_agent(safety_opt, requireModelExists=True)
    agent.model.eval()
    return agent


def update_history(agent, context: List[str]):
    observation = {"text_vec": None}
    agent.reset()
    for utterance_idx, utterance in enumerate(context):
        observe_input = {"text": utterance, "episode_done": False}
        is_self_utterance = (len(context) - utterance_idx) % 2 == 0
        if is_self_utterance:
            # Initial utterance case
            if agent.observation is None:
                agent.history.add_reply(utterance)
            else:
                agent.self_observe(observe_input)
        else:
            observation = agent.observe(observe_input)

    if observation["text_vec"] is None:
        observation = agent.observe({"text": " ", "episode_done": False})
    return observation["text_vec"]


def get_embeddings(agent, contexts: List[List[str]]):
    logging.disable()
    with torch.no_grad():
        text_vecs = [update_history(agent, context) for context in contexts]
        padded_text_vecs = agent._pad_tensor(text_vecs)[0].to("cuda:0")
        batch = Batch(text_vec=padded_text_vecs)
        _, _context_embeddings = agent.model.encode_context_memory(
            context_w=batch.text_vec, memories_w=None, context_segments=None)
    logging.enable()
    return _context_embeddings


def get_candidate_embeddings(agent, candidates: List[str]):
    with torch.no_grad():
        logging.disable()
        sys.stderr = open(os.devnull, "w")
        candidate_vecs = agent._make_candidate_vecs(candidates)
        candidate_embs = agent._make_candidate_encs(candidate_vecs.to("cuda:0"))
        logging.enable()
        sys.stderr = sys.__stderr__
    return candidate_embs


def build_context_embeddings(agent, exemplars, batch_size=32):
    context_embeddings = []
    for start_idx in range(0, len(exemplars), batch_size):
        end_idx = min(len(exemplars), start_idx + batch_size)
        context_embeddings.append(
            get_embeddings(
                agent,
                [[exemplar.context] for exemplar in exemplars[start_idx:end_idx]]
            )
        )

    context_embeddings = torch.cat(context_embeddings, dim=0)
    return context_embeddings


def read_exemplars(exemplar_tsv_path, use_k_sentences=-1, style=None, delimiter="\t", use_gold_q=False):
    df = pd.read_csv(exemplar_tsv_path, delimiter=delimiter).dropna()
    if style:
        df = df.loc[df["style"] == style]
        if not use_gold_q:
            df["context"] = "DUMMY"
    df = df.rename(columns={"generated_text": "response"})
    exemplars = [row for _, row in df.iterrows()]
    for exp in exemplars:
        if exp.response[0] == '"' and exp.response[-1] == '"':
            exp.response = exp.response[1:-1]
    if use_k_sentences > -1:
        exemplars = exemplars[:use_k_sentences]  # use first k sentences
    return exemplars


def setup_exemplars(agent, exemplar_tsv_path, batch_size=32, use_k_sentences=-1, style=None):
    exemplars = read_exemplars(exemplar_tsv_path, use_k_sentences, style)
    context_embeddings = build_context_embeddings(agent, exemplars, batch_size)
    return exemplars, context_embeddings


def get_reference_exemplars(agent, query_context, exemplars, context_embeddings, max_num_exemplars):
    with torch.no_grad():
        query_embedding = get_embeddings(agent, [query_context])
        scores = torch.mm(query_embedding, context_embeddings.transpose(0, 1)).squeeze(0)
        scores = scores.cpu().numpy()

    reference_exemplars = [
        {"score": float(scores[idx]), **exemplars[idx].to_dict()}
        for idx in np.argsort(scores)[::-1][:max_num_exemplars]
    ]
    reference_exemplars = sorted(reference_exemplars, key=lambda x: -x["score"])
    return reference_exemplars


def get_ranker_top1(agent, context, candidates, print_candidates=False):
    with torch.no_grad():
        context_embedding = get_embeddings(agent, [context])
        candidate_embeddings = get_candidate_embeddings(agent, [cand["text"] for cand in candidates])
        scores = torch.mm(context_embedding, candidate_embeddings.transpose(0, 1)).squeeze(0)
        scores = scores.cpu().numpy()

    if print_candidates:
        for score, candidate in zip(scores, candidates):
            candidate["score"] = score
        print_candidates = sorted(candidates, key=lambda x: -x["score"])
        print(print_candidates)

    return candidates[np.argmax(scores)]


def select_response(agent, context, candidates, strategy_name):
    if strategy_name == "random":
        candidate = random.choice(candidates)
    elif strategy_name == "top1":
        candidate = max(candidates, key=lambda x: x["score"])
    elif strategy_name == "ranker":
        candidate = get_ranker_top1(agent, context, candidates)
    else:
        raise ValueError(f"Not a proper selection strategy.")

    return candidate["text"]


def get_pseudo_context_embeddings(agent, pseudo_contexts):
    batch_size = 1024
    context_embeddings = []
    for start_idx in tqdm(range(0, len(pseudo_contexts), batch_size)):
        end_idx = min(len(pseudo_contexts), start_idx + batch_size)
        batch = pseudo_contexts[start_idx: end_idx]
        contexts = [[context] for context in batch]
        context_embedding = get_embeddings(agent, contexts)
        context_embeddings.append(context_embedding)
    context_embeddings = torch.cat(context_embeddings)
    assert len(pseudo_contexts) == len(context_embeddings)
    return context_embeddings


def build_faiss_index(fixed_candidate_vecs: np.ndarray):
    print("Started building faiss index.")
    num_embeddings, index_dim = fixed_candidate_vecs.shape
    cpu_faiss_index = faiss.IndexFlatIP(index_dim)
    gpu_faiss_index = cpu_faiss_index
    # gpu_faiss_index = faiss.index_cpu_to_all_gpus(cpu_faiss_index)
    gpu_faiss_index.add(fixed_candidate_vecs)
    print("Finished building faiss index.")
    return gpu_faiss_index


def get_reference_exemplars_using_answer(agent, query_context, exemplars, candidate_embeddings, max_num_exemplars):
    with torch.no_grad():
        query_embedding = get_embeddings(agent, [query_context])
        scores = torch.mm(query_embedding, candidate_embeddings.transpose(0, 1)).squeeze(0)
        scores = scores.cpu().numpy()

    reference_exemplars = [
        {"score": float(scores[idx]), **exemplars[idx].to_dict()}
        for idx in np.argsort(scores)[::-1][:max_num_exemplars]
    ]
    reference_exemplars = sorted(reference_exemplars, key=lambda x: -x["score"])
    return reference_exemplars
