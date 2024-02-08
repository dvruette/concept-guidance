import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concept_guidance.activations import compute_activations
from concept_guidance.data.open_assistant import get_open_assistant_messages
from concept_guidance.models.difference_in_means import DiMProbe
from concept_guidance.models.logistic import LogisticProbe
from concept_guidance.models.pca import PCAProbe

logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--probe", type=str, default="dim", choices=["dim", "logistic", "pca"])
    parser.add_argument("--ctx_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=512)
    args = parser.parse_args()
    return args


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples = get_open_assistant_messages(label_key="humor", max_messages=args.num_samples)

    # Compute model activations
    activations, labels = compute_activations(model, tokenizer, examples, ctx_len=args.ctx_len)

    # Train a probe on the activations
    if args.probe == "dim":
        probe = DiMProbe()
    elif args.probe == "logistic":
        probe = LogisticProbe()
    elif args.probe == "pca":
        probe = PCAProbe()
    else:
        raise ValueError(f"Unknown probe: {args.probe}")
    probe.fit(activations, labels)

    # To save the probe
    probe.save("humor.safetensors")


if __name__ == "__main__":
    args = parse_args()
    main(args)
