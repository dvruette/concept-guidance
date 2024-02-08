import copy

import torch
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig

from concept_guidance.activations import get_tokens


@torch.no_grad()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    system_prompt: str = "",
    max_input_len: int = 1024,
    max_new_tokens: int = 196,
    truncation: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    stop_at_newline: bool = False,
):
    device = next(model.parameters()).device

    seen_prompts = set()
    deduped_messages = []
    for msg in messages:
        if msg["prompt"] in seen_prompts:
            continue
        seen_prompts.add(msg["prompt"])
        deduped_messages.append(msg)

    config = GenerationConfig(
        do_sample=do_sample,
        num_return_sequences=1,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=[tokenizer.eos_token_id] + (tokenizer.encode("\n", add_special_tokens=False) if stop_at_newline else []),
        pad_token_id=tokenizer.eos_token_id,
    )

    completions = []
    for message in tqdm.tqdm(deduped_messages, desc="Generating", smoothing=0.01):
        msg = {
            "prompt": message["prompt"],
            "completion": "",
            "conversation_history": message.get("conversation_history", None)
        }
        inputs, _ = get_tokens(
            tokenizer, msg,
            system_prompt=system_prompt,
            max_length=max_input_len,
            truncation=truncation,
        )
        inputs = inputs.to(device)

        # Removes last spaces and eos. Valid also for non-chat models
        input_ids = inputs["input_ids"][:, :-2]
        attention_mask = inputs["attention_mask"][:, :-2]

        outputs = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
        )

        if input_ids[0, 0] == tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]
            outputs = outputs[:, 1:]

        prompt = tokenizer.decode(input_ids[0])
        completion = tokenizer.decode(outputs[0, len(input_ids[0]):-1])
        
        result = copy.deepcopy(message)
        result["user_prompt"] = message["prompt"]
        result["prompt"] = prompt.strip()
        result["completion"] = completion.strip()
        result.pop("conversation_history", None)
        completions.append(result)

    return completions
