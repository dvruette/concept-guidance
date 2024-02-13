import argparse
import logging
from threading import Thread

import time
import torch
import gradio as gr
from concept_guidance.chat_template import DEFAULT_CHAT_TEMPLATE
from concept_guidance.patching import patch_model, load_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer, Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

MODEL_CONFIGS = {
    "Llama-2-7b-chat-hf": {
        "identifier": "meta-llama/Llama-2-7b-chat-hf",
        "dtype": torch.float16 if device.type == "cuda" else torch.float32,
        "guidance_interval": [-16.0, 16.0],
        "default_guidance_scale": 8.0,
        "min_guidance_layer": 16,
        "max_guidance_layer": 32,
        "default_concept": "humor",
        "concepts": ["humor", "creativity", "quality", "truthfulness", "compliance"],
    },
    "Mistral-7B-Instruct-v0.1": {
        "identifier": "mistralai/Mistral-7B-Instruct-v0.1",
        "dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "guidance_interval": [-128.0, 128.0],
        "default_guidance_scale": 48.0,
        "min_guidance_layer": 8,
        "max_guidance_layer": 32,
        "default_concept": "humor",
        "concepts": ["humor", "creativity", "quality", "truthfulness", "compliance"],
    },
}

def load_concept_vectors(model, concepts):
    return {concept: load_weights(f"trained_concepts/{model}/{concept}.safetensors") for concept in concepts}

def load_model(model_name):
    config = MODEL_CONFIGS[model_name]
    model = AutoModelForCausalLM.from_pretrained(config["identifier"], torch_dtype=config["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(config["identifier"])
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return model, tokenizer

CONCEPTS = ["humor", "creativity", "quality", "truthfulness", "compliance"]
CONCEPT_VECTORS = {model_name: load_concept_vectors(model_name, CONCEPTS) for model_name in MODEL_CONFIGS}
MODELS = {model_name: load_model(model_name) for model_name in MODEL_CONFIGS}


def history_to_conversation(history):
    conversation = Conversation()
    for prompt, completion in history:
        conversation.add_message({"role": "user", "content": prompt})
        if completion is not None:
            conversation.add_message({"role": "assistant", "content": completion})
    return conversation



def set_defaults(model_name):
    config = MODEL_CONFIGS[model_name]
    return (
        model_name,
        gr.update(choices=config["concepts"], value=config["concepts"][0]),
        gr.update(minimum=config["guidance_interval"][0], maximum=config["guidance_interval"][1], value=config["default_guidance_scale"]),
        gr.update(value=config["min_guidance_layer"]),
        gr.update(value=config["max_guidance_layer"]),
    )

def add_user_prompt(user_message, history):
    if history is None:
        history = []
    history.append([user_message, None])
    return history

@torch.no_grad()
def generate_completion(
    history,
    model_name,
    concept,
    guidance_scale=4.0,
    min_guidance_layer=16,
    max_guidance_layer=32,
    temperature=0.0,
    repetition_penalty=1.2,
    length_penalty=1.2,
):
    start_time = time.time()
    logger.info(f" --- Starting completion ({model_name}, {concept=}, {guidance_scale=}, {min_guidance_layer=}, {temperature=})") 
    logger.info(" User: " + repr(history[-1][0]))
    
    # move all other models to CPU
    for name, (model, _) in MODELS.items():
        if name != model_name:
            model.to("cpu")
    torch.cuda.empty_cache()
    # load the model
    model, tokenizer = MODELS[model_name]
    model = model.to(device, non_blocking=True)

    concept_vector = CONCEPT_VECTORS[model_name][concept]
    guidance_layers = list(range(int(min_guidance_layer) - 1, int(max_guidance_layer)))
    patch_model(model, concept_vector, guidance_scale=guidance_scale, guidance_layers=guidance_layers)
    pipe = pipeline("conversational", model=model, tokenizer=tokenizer, device=device)
    
    conversation = history_to_conversation(history)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        max_new_tokens=512,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        streamer=streamer,
        temperature=temperature,
        do_sample=(temperature > 0)
    )
    thread = Thread(target=pipe, args=(conversation,), kwargs=generation_kwargs, daemon=True)
    thread.start()

    history[-1][1] = ""
    for token in streamer:
        history[-1][1] += token
        yield history
    logger.info(" Assistant: " + repr(history[-1][1]))
    
    time_taken = time.time() - start_time
    logger.info(f" --- Completed (took {time_taken:.1f}s)")
    return history


class ConceptGuidanceUI:
    def __init__(self):
        model_names = list(MODEL_CONFIGS.keys())
        default_model = model_names[0]
        default_config = MODEL_CONFIGS[default_model]
        default_concepts = default_config["concepts"]

        saved_input = gr.State("")

        with gr.Row(elem_id="concept-guidance-container"):
            with gr.Column(scale=1, min_width=256):
                model_dropdown = gr.Dropdown(model_names, value=default_model, label="Model")
                concept_dropdown = gr.Dropdown(default_concepts, value=default_concepts[0], label="Concept")
                guidance_scale = gr.Slider(*default_config["guidance_interval"], value=default_config["default_guidance_scale"], label="Guidance Scale")
                min_guidance_layer = gr.Slider(1.0, 32.0, value=16.0, step=1.0, label="First Guidance Layer")
                max_guidance_layer = gr.Slider(1.0, 32.0, value=32.0, step=1.0, label="Last Guidance Layer")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Temperature")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.01, label="Repetition Penalty")
                length_penalty = gr.Slider(0.0, 2.0, value=1.2, step=0.01, label="Length Penalty")

            with gr.Column(scale=3, min_width=512):
                chatbot = gr.Chatbot(scale=1, height=200)

                with gr.Row():
                    self.retry_btn = gr.Button("🔄 Retry", size="sm")
                    self.undo_btn = gr.Button("↩️ Undo", size="sm")
                    self.clear_btn = gr.Button("🗑️ Clear", size="sm")
                
                with gr.Group():
                    with gr.Row():
                        prompt_field = gr.Textbox(placeholder="Type a message...", show_label=False, label="Message", scale=7, container=False)
                        self.submit_btn = gr.Button("Submit", variant="primary", scale=1, min_width=150)
                        self.stop_btn = gr.Button("Stop", variant="secondary", scale=1, min_width=150, visible=False)

        generation_args = [
            model_dropdown,
            concept_dropdown,
            guidance_scale,
            min_guidance_layer,
            max_guidance_layer,
            temperature,
            repetition_penalty,
            length_penalty,
        ]

        model_dropdown.change(set_defaults, [model_dropdown], [model_dropdown, concept_dropdown, guidance_scale, min_guidance_layer, max_guidance_layer], queue=False)

        submit_triggers = [prompt_field.submit, self.submit_btn.click]
        submit_event = gr.on(
            submit_triggers, self.clear_and_save_input, [prompt_field], [prompt_field, saved_input], queue=False
        ).then(
            add_user_prompt, [saved_input, chatbot], [chatbot], queue=False
        ).then(
            generate_completion,
            [chatbot] + generation_args,
            [chatbot],
            concurrency_limit=1,
        )
        self.setup_stop_events(submit_triggers, submit_event)

        retry_triggers = [self.retry_btn.click]
        retry_event = gr.on(
            retry_triggers, self.delete_prev_message, [chatbot], [chatbot, saved_input], queue=False
        ).then(
            add_user_prompt, [saved_input, chatbot], [chatbot], queue=False
        ).then(
            generate_completion,
            [chatbot] + generation_args,
            [chatbot],
            concurrency_limit=1,
        )
        self.setup_stop_events(retry_triggers, retry_event)

        self.undo_btn.click(
            self.delete_prev_message, [chatbot], [chatbot, saved_input], queue=False
        ).then(
            lambda x: x, [saved_input], [prompt_field]
        )
        self.clear_btn.click(lambda: [None, None], None, [chatbot, saved_input], queue=False)

    def clear_and_save_input(self, message):
        return "", message
    
    def delete_prev_message(self, history):
        message, _ = history.pop()
        return history, message or ""

    def setup_stop_events(self, event_triggers, event_to_cancel):
        if self.submit_btn:
            for event_trigger in event_triggers:
                event_trigger(
                    lambda: (
                        gr.Button(visible=False),
                        gr.Button(visible=True),
                    ),
                    None,
                    [self.submit_btn, self.stop_btn],
                    show_api=False,
                    queue=False,
                )
            event_to_cancel.then(
                lambda: (gr.Button(visible=True), gr.Button(visible=False)),
                None,
                [self.submit_btn, self.stop_btn],
                show_api=False,
                queue=False,
            )

        self.stop_btn.click(
            None,
            None,
            None,
            cancels=event_to_cancel,
            show_api=False,
        )

css = """
#concept-guidance-container {
    flex-grow: 1;
}
""".strip()

with gr.Blocks(title="Concept Guidance", fill_height=True, css=css) as demo:
    ConceptGuidanceUI()

demo.queue()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(share=args.share)