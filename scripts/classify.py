import argparse
import glob
import json
import re
from pathlib import Path

import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

import concept_guidance.eval.toxic_completions as toxic_completions
import concept_guidance.eval.truthfulqa as truthfulqa
import concept_guidance.eval.open_assistant as open_assistant

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
    parser.add_argument("--concept", type=str, default="compliance", choices=CONCEPT_MAP.keys())
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def load_file(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for x in data:
        x["prompt"] = re.sub(r"\[/?INST\]", "", x["prompt"]).strip()
    return data


def classify(data: list[dict], concept: str, llm=None):
    dataset, label_key = CONCEPT_MAP[concept]
    if dataset == "toxic-completions":
        return toxic_completions.classify(llm, data)
    elif dataset == "truthfulqa":
        return truthfulqa.classify(data, models=llm)
    elif dataset == "open-assistant":
        return open_assistant.classify(llm, data, label_key=label_key)
    else:
        raise ValueError(f"Invalid concept: {concept}")


def compute_metrics(data, concept: str):
    dataset, _ = CONCEPT_MAP[concept]
    if dataset == "toxic-completions":
        return toxic_completions.compute_metrics(data)
    elif dataset == "truthfulqa":
        return truthfulqa.compute_metrics(data)
    elif dataset == "open-assistant":
        return open_assistant.compute_metrics(data)
    else:
        raise ValueError(f"Invalid concept: {concept}")


def main(args):
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise ValueError(f"Invalid input folder: {input_path}")
    
    all_files = sorted(glob.glob(str(input_path / "**" / "completions.json")))
    if len(all_files) == 0:
        raise ValueError(f"No completions found in {input_path}")

    if args.concept in ["appropriateness", "compliance", "humor", "creativity", "quality"]:
        llm = LLM("mistralai/Mistral-7B-v0.1", max_model_len=2048, download_dir=args.cache_dir)
    elif args.concept in ["truthfulness"]:
        model_name = f"sanagnos/Mistral-7B-v0.1-truthfulqa-info"

        info_tokenizer = AutoTokenizer.from_pretrained(model_name)
        info_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=0,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            cache_dir=args.cache_dir
        )
        info_model.eval()

        model_name = f"sanagnos/Mistral-7B-v0.1-truthfulqa-truth"
        
        truth_tokenizer = AutoTokenizer.from_pretrained(model_name)
        truth_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=0,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            cache_dir=args.cache_dir
        )
        truth_model.eval()

        llm = {
            'info_tokenizer': info_tokenizer,
            'info_model': info_model,
            'truth_tokenizer': truth_tokenizer,
            'truth_model': truth_model
        }
    else: 
        raise ValueError(f"Invalid concept: {args.concept}")
    
    if args.output_dir is None:
        output_path = input_path
    else:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

    for input_file in tqdm.tqdm(all_files, disable=len(all_files) == 1, desc="Files"):
        input_file = Path(input_file)
        curr_path = input_file.parent

        try:
            with open(curr_path / "metrics.json", "r") as f:
                metrics = json.load(f)
        except OSError:
            metrics = {}

        data = load_file(input_file)
        data = classify(data, args.concept, llm=llm)
        metrics.update(compute_metrics(data, args.concept))

        curr_output = Path(str(curr_path).replace(str(input_path), str(output_path)))

        with open(curr_output / "completions.json", "w") as f:
            json.dump(data, f, indent=4)

        with open(curr_output / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if len(all_files) == 1:
            print(f"Saved classified completions to {curr_output / 'completions.json'}")
            print(f"Saved metrics to {curr_output / 'metrics.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
