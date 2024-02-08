import copy
from typing import Literal

import torch
import tqdm
import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from baukit import TraceDict


def get_tokens(
    tokenizer: PreTrainedTokenizerBase,
    message: dict[str, str],
    system_prompt: str = "",
    max_length: int = 1024,
    max_assistant_tokens: int = 32,
    truncation: bool = True,
):
    history = [] if "conversation_history" not in message or message["conversation_history"] is None else copy.deepcopy(message["conversation_history"])
    history.extend([
        {"role": "user", "content": message["prompt"]},
        {"role": "assistant", "content": message["completion"]},
    ])
    if system_prompt:
        history = [{"role": "system", "content": system_prompt}] + history

    prompt = tokenizer.apply_chat_template(history, tokenize=False)
    if message["completion"]:
        # TODO this does not work for short answers, e.g. for MMLU ... 
        # prefix = prompt.split(message["answer"])[0].strip()
        history_ = copy.deepcopy(history)
        history_.pop()
        history_.append({"role": "assistant", "content": ""})
        prompt_ = tokenizer.apply_chat_template(history_, tokenize=False)
        num_prefix_tokens = len(tokenizer(prompt_, add_special_tokens=False)["input_ids"]) - 2 # remove the last two tokens for the empty answer TODO does this hold for all tokenizers?? (I think so)
    else:
        prefix = prompt.strip()
        num_prefix_tokens = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)

    # truncate right (assistant response)
    if truncation and max_assistant_tokens > 0:
        for key in inputs:
            inputs[key] = inputs[key][:, :num_prefix_tokens + max_assistant_tokens]
    num_completion_tokens = len(inputs["input_ids"][0]) - num_prefix_tokens

    # truncate left (history)
    if truncation and max_length > 0:
        for key in inputs:
            inputs[key] = inputs[key][:, -max_length:]
        num_completion_tokens = min(max_length, num_completion_tokens)

    return inputs, num_completion_tokens


RepresentationType = Literal[
    "hiddens", "pre-attn", "queries", "keys", "values", "heads", "mlp", "post-attn"
]

@torch.no_grad()
def compute_activations(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    messages: list[dict[str, str]],
    system_prompt: str = "",
    representation: RepresentationType = "pre-attn",  
    ctx_len: int = 16,
    max_assistant_tokens: int = 32,
    max_input_len=1024, 
    truncation=True, 
    max_messages=-1,
    output_dtype=torch.float32,
    return_metrics: bool = False,
) -> (list[NDArray], NDArray, dict):
    device = next(model.parameters()).device
    activations = []
    labels = []

    metrics = {
        "ppl": [],
        "ce_loss": [],
    }

    trace_modules = []
    if representation == "pre-attn":
        target_key = "input_layernorm"
        error_msg = f"Pre-norm not found in {model.__class__.__name__}"
    elif representation == "queries":
        target_key = "self_attn.q_proj"
        error_msg = f"Query not found in {model.__class__.__name__}"
    elif representation in ("heads", "post-attn"):
        target_key = "self_attn"
        error_msg = f"Self-attention not found in {model.__class__.__name__}"
    elif representation == "mlp":
        target_key = "mlp"
        error_msg = f"MLP not found in {model.__class__.__name__}"
    else:
        target_key = None
        error_msg = None

    if target_key is not None:
        for i, layer in enumerate(model.model.layers):
            found = False
            for name, _ in layer.named_modules():
                if target_key in name:
                    trace_modules.append(f"model.layers.{i}.{name}")
                    found = True
                    break
            if not found:
                raise ValueError(error_msg)
            
    if max_messages < 0:
        max_messages = len(messages)

    with TraceDict(model, trace_modules) as trace:
        with tqdm.tqdm(total=min(len(messages), max_messages), desc="Activations", smoothing=0.01) as pbar:
            for msg in messages:
                if len(activations) >= max_messages:
                    break

                inputs, num_completion_tokens = get_tokens(
                    tokenizer, msg,
                    system_prompt=system_prompt,
                    max_length=max_input_len,
                    max_assistant_tokens=max_assistant_tokens,
                    truncation=truncation,
                )
                inputs = inputs.to(device)
                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=True)

                if return_metrics:
                    if len(outputs.logits[0, -num_completion_tokens:]) > 1:
                        ce_loss = F.cross_entropy(
                            outputs.logits[0, -num_completion_tokens:-1],
                            inputs["input_ids"][0, -num_completion_tokens+1:],
                            reduction="mean",
                        ).to(torch.float32)
                        ppl = torch.exp(ce_loss)
                        metrics["ppl"].append(ppl.item())
                        metrics["ce_loss"].append(ce_loss.item())
                    else:
                        metrics["ppl"].append(np.nan)
                        metrics["ce_loss"].append(np.nan)

                if representation == "hiddens":
                    reps = torch.cat(outputs.hidden_states[1:]).squeeze(1)
                elif representation == "pre-attn":
                    hiddens = [trace[name].output for name in trace_modules]
                    reps = torch.cat(hiddens, dim=0)
                elif representation == "queries":
                    queries = [trace[name].output for name in trace_modules]
                    reps = torch.cat(queries, dim=0)
                elif representation == "keys":
                    reps = torch.cat([x[0] for x in outputs.past_key_values]).squeeze(1).transpose(1, 2)
                    reps = reps.reshape(reps.shape[:2] + (-1,))
                elif representation == "values":
                    reps = torch.cat([x[1] for x in outputs.past_key_values]).squeeze(1).transpose(1, 2)
                    reps = reps.reshape(reps.shape[:2] + (-1,))
                elif representation == "heads":
                    num_heads = model.config.num_attention_heads
                    hiddens = [trace[name].output[0] for name in trace_modules]
                    hiddens = torch.cat(hiddens, dim=0)
                    heads = hiddens.reshape(*hiddens.shape[:2], num_heads, -1)
                    reps = heads
                elif representation == "post-attn":
                    post_attn = [trace[name].output[0] for name in trace_modules]
                    reps = torch.cat(post_attn, dim=0)
                elif representation == "mlp":
                    mlp = [trace[name].output for name in trace_modules]
                    reps = torch.cat(mlp, dim=0)
                else:
                    raise ValueError(f"Unknown representation: {representation}")
                
                reps = reps[:, -num_completion_tokens:-1] 

                if ctx_len > 0:
                    reps = reps[:, :ctx_len]

                activations.append(reps.cpu().to(output_dtype).transpose(0, 1))
                labels.append(msg["label"])
                pbar.update(1)

    labels = torch.tensor(labels, dtype=torch.long)
    if return_metrics:
        return activations, labels, metrics
    return activations, labels
