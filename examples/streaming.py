import argparse
import logging

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from concept_guidance.patching import load_weights, patch_model
from concept_guidance.chat_template import DEFAULT_CHAT_TEMPLATE

logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Hugging Face model name")
    parser.add_argument("--concept", type=str, default="trained_concepts/Llama-2-7b-chat-hf/humor.safetensors", help="Path to trained concept vector")
    parser.add_argument("--guidance_scale", type=float, default=32.0)
    parser.add_argument("--guidance_layers", type=int, nargs="+", default=range(8, 32))
    args = parser.parse_args()
    return args


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    concept_vectors = load_weights(args.concept_vectors)
    patch_model(model, concept_vectors, guidance_scale=args.guidance_scale, guidance_layers=args.guidance_layers)

    pipe = pipeline("conversational", model=model, tokenizer=tokenizer)
    conversation = transformers.Conversation()

    print(f"==== Concept-Guided Chat with {args.model} ({args.concept_vectors}) ====")
    print("type (q) to quit")
    while True:
        prompt = input("User: ")
        if prompt == "q":
            exit()
        conversation.add_user_input(prompt)
        print("Assistant: ", end="")
        conversation = pipe(conversation, max_new_tokens=256, repetition_penalty=1.2, streamer=streamer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
