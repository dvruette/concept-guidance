import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from concept_guidance.activations import compute_activations
from concept_guidance.metrics import layer_wise_accuracy
from concept_guidance.utils import parse_dtype, get_data, train_probe
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
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--concept", type=str, default="compliance", choices=CONCEPT_MAP.keys())
    parser.add_argument("--probe", type=str, default="dim", choices=["dim", "logistic", "pca"])
    parser.add_argument("--representation", type=str, default="pre-attn")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--do_few_shot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ctx_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--max_num_generate", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_flash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    model_name = args.model
    probe = args.probe
    do_few_shot = args.do_few_shot
    dataset, label_key = CONCEPT_MAP[args.concept]

    with open(output_path / "config.json", "w") as f:
        json.dump(dict(vars(args)), f, indent=4)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    device_map = "auto" if device.type == "cpu" else (device.index or 0)

    print(f"Using device: {device} ({device_map=})")


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


    train_messages, val_messages, _, num_train = get_data(
        dataset=dataset,
        num_samples=args.num_samples,
        max_num_generate=args.max_num_generate,
        do_few_shot=do_few_shot,
        label_key=label_key,
        cache_dir=args.cache_dir,
    )

    activations, labels = compute_activations(
        llm, tokenizer, train_messages + val_messages,
        system_prompt=args.system_prompt,
        representation=args.representation,
        max_messages=args.num_samples,
        ctx_len=args.ctx_len,
    )
    train_labels = labels[:num_train]
    val_labels = labels[num_train:]
    train_xs = [x for x in activations[:num_train]]
    val_xs = [x for x in activations[num_train:]]

    model = train_probe(probe, train_xs, train_labels, device=device)

    train_preds = model.predict(train_xs)
    val_preds = model.predict(val_xs)

    train_accs = layer_wise_accuracy(train_preds, train_labels)
    val_accs = layer_wise_accuracy(val_preds, val_labels)

    print(f"Model: {model_name} (probe={args.probe})")
    print(f"Train Acc: mean={train_accs.mean():.4g} - max={train_accs.max():.4g}")
    print(f"Val Acc: mean={val_accs.mean():.4g} - max={val_accs.max():.4g}")
    
    model_file = output_path / "weights.safetensors"
    model.save(model_file)
    print(f"Saved probe weights to {model_file}")
    
    metrics = {
        "model": model_name,
        "concept": args.concept,
        "label_key": label_key,
        "probe": args.probe,
        "representation": args.representation,
        "ctx_len": args.ctx_len,
        "mean_train_acc": float(train_accs.mean()),
        "mean_val_acc": float(val_accs.mean()),
        "max_train_acc": float(train_accs.max()),
        "max_val_acc": float(val_accs.max()),
        "train_accs": train_accs.tolist(),
        "val_accs": val_accs.tolist(),
    }

    outfile = output_path / "metrics.json"
    with open(outfile, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {outfile}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
