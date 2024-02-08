import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from concept_guidance.activations import compute_activations
from concept_guidance.data.open_assistant import get_open_assistant_messages
from concept_guidance.generation import generate_completions
from concept_guidance.patching import load_weights, patch_model, unpatch_model
from concept_guidance.utils import parse_dtype, get_data
from concept_guidance.chat_template import DEFAULT_CHAT_TEMPLATE


CONCEPT_MAP = {
    "compliance": ("toxic-completions", None),
    "appropriateness": ("toxic-completions", None),
    "truthfulness": ("truthfulqa", None),
    "humor": ("open-assistant", "humor"),
    "creativity": ("open-assistant", "creativity"),
    "quality": ("open-assistant", "quality"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None, help="If not provided, will save in the same folder as input_dir")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--concept", type=str, default="compliance", choices=CONCEPT_MAP.keys())
    parser.add_argument("--representation", type=str, default="pre-attn")
    parser.add_argument("--guidance_scale", type=float, nargs="+", default=[4])
    parser.add_argument("--guidance_top_k", type=int, default=16)
    parser.add_argument("--guide_only_last", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--max_num_generate", type=int, default=64)
    parser.add_argument("--is_chat_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_few_shot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_flash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise ValueError(f"Invalid input folder: {input_path}")
    
    if not (input_path / "weights.safetensors").exists():
        raise ValueError(f"Missing weights file: {input_path / 'weights.safetensors'}")
    
    if not (input_path / "metrics.json").exists():
        raise ValueError(f"Missing metrics file (required for selecting top-k layers): {input_path / 'metrics.json'}")

    if args.output_dir is None:
        output_path = input_path
    else:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

    model_name = args.model
    do_few_shot = args.do_few_shot
    dataset, label_key = CONCEPT_MAP[args.concept]

    with open(output_path / "guidance_config.json", "w") as f:
        json.dump(dict(vars(args)), f, indent=4)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    device_map = "auto" if device.type == "cpu" else (device.index or 0)
    print(f"Using device: {device} ({device_map=})")

    ppl_dataset = get_open_assistant_messages(max_messages=256, label_key="quality")
    ppl_dataset = [x for x in ppl_dataset if x["label"] == 1]
    print(f"Loaded {len(ppl_dataset)} messages for PPL computation")

    has_flash_attn = False
    if args.use_flash:
        try:
            import flash_attn
            has_flash_attn = True
        except ImportError:
            pass

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=parse_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="flash_attention_2" if has_flash_attn else None,
        cache_dir=args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    _, _, completion_messages, _ = get_data(
        dataset=dataset,
        num_samples=args.num_samples,
        max_num_generate=args.max_num_generate,
        do_few_shot=do_few_shot,
        label_key=label_key,
        cache_dir=args.cache_dir,
    )

    concept_vectors = load_weights(input_path / "weights.safetensors", device=device)
    
    with open(input_path / "metrics.json", "r") as f:
        metrics = json.load(f)

    train_accs = np.array(metrics["train_accs"])

    thresh_val = np.sort(train_accs.flatten())[-args.guidance_top_k]
    ids = np.where(train_accs >= thresh_val)
    guidance_layers = list(zip(*ids))

    for guidance_scale in tqdm.tqdm(args.guidance_scale, disable=len(args.guidance_scale) == 1):
        patch_model(
            llm, concept_vectors,
            representation=args.representation,
            guidance_scale=guidance_scale,
            guidance_layers=guidance_layers,
            only_last=args.guide_only_last,
        )

        _, _, patched_metrics = compute_activations(
            llm, tokenizer, ppl_dataset,
            system_prompt=args.system_prompt,
            representation=args.representation,
            max_messages=args.num_samples,
            max_assistant_tokens=-1,
            return_metrics=True,
        )
            
        completions = generate_completions(
            llm, tokenizer, completion_messages,
            system_prompt=args.system_prompt,
            do_sample=False,
            stop_at_newline=False if args.is_chat_model else True,
        )

        unpatch_model(llm)

        metrics = {
            "guidance_scale": guidance_scale,
            "guidance_top_k": args.guidance_top_k,
            "guide_only_last": args.guide_only_last,
            "ppl": float(np.nanmean(patched_metrics["ppl"])),
        }

        curr_path = output_path / f"alpha={guidance_scale}"
        curr_path.mkdir(exist_ok=True, parents=True)

        with open(curr_path / "completions.json", "w") as f:
            json.dump(completions, f, indent=4)

        with open(curr_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if len(args.guidance_scale) == 1:
            print(f"Saved completions to {curr_path / 'completions.json'}")
            print(f"Saved metrics to {curr_path / 'metrics.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
