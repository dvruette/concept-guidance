from collections import Counter
import random

import torch

from concept_guidance.data.open_assistant import get_open_assistant_messages
from concept_guidance.data.truthfulqa import get_truthfulqa_messages
from concept_guidance.data.toxic_completions import get_toxic_completions_messages
from concept_guidance.models.base import BaseProbe
from concept_guidance.models.difference_in_means import DiMProbe
from concept_guidance.models.logistic import LogisticProbe
from concept_guidance.models.pca import PCAProbe


def parse_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def get_data(
    dataset: str,
    num_samples: int,
    max_num_generate: int = 0,
    do_few_shot: bool = False,
    label_key: str | None = None,
    cache_dir: str | None = None,
):
    if dataset == "truthfulqa":
        messages = get_truthfulqa_messages(max_messages=num_samples + 2 * max_num_generate, do_few_shot=do_few_shot, cache_dir=cache_dir)
    elif dataset == "open-assistant":
        messages = get_open_assistant_messages(max_messages=num_samples + 2 * max_num_generate, label_key=label_key, do_few_shot=do_few_shot, cache_dir=cache_dir)
    elif dataset == "toxic-completions":
        messages = get_toxic_completions_messages(max_messages=num_samples + 2 * max_num_generate, do_few_shot=do_few_shot, cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    prompt_counts = Counter([message["prompt"] for message in messages])
    all_prompts = list(prompt_counts.keys())
    random.shuffle(all_prompts)

    num_train = int(0.8*num_samples)
    num_val = num_samples - num_train
    train_count = 0
    train_prompts, val_prompts = [], []
    for prompt in all_prompts:
        if train_count < num_train:
            train_prompts.append(prompt)
            train_count += prompt_counts[prompt]
        else:
            val_prompts.append(prompt)

    train_messages = [message for message in messages if message["prompt"] in train_prompts]
    val_messages = [message for message in messages if message["prompt"] in val_prompts]

    completion_messages = []
    seen_prompts = set()
    for message in val_messages:
        if message["prompt"] not in seen_prompts:
            completion_messages.append(message)
            seen_prompts.add(message["prompt"])
        if len(completion_messages) >= max_num_generate:
            break
    val_messages = val_messages[:num_val]
    
    return train_messages, val_messages, completion_messages, num_train


def train_probe(probe, train_xs, train_labels, device="cpu") -> BaseProbe:
    if probe == "dim":
        model = DiMProbe(normalize=True, device=device)
        model.fit(train_xs, train_labels)
    elif probe == "logistic":
        model = LogisticProbe(normalize=True, l2_penalty=1.0)
        model.fit(train_xs, train_labels)
    elif probe == "pca":
        model = PCAProbe(normalize=False, device=device)
        model.fit(train_xs, train_labels)
    else:
        raise NotImplementedError(f"Invalid probe: {probe}")
    return model
