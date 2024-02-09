# Concept Guidance

Code accompanying the paper "A Language Model's Guide Through Latent Space".

Paper: [COMING SOON]

Demo: [COMING SOON]


## Installation

```bash
pip install git+https://github.com/dvruette/concept-guidance.git
```

## Usage

### Concept-Guided Generation

To use the concept vectors for concept-guided generation, we patch the model with the learned concept vectors.
Guidance strength is controlled by the `guidance_scale` parameter, and which layers to apply guidance to is controlled by the `guidance_layers` parameter.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from concept_guidance.patching import load_weights, patch_model, unpatch_model

model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Load the probe
concept_vectors = load_weights("concept.safetensors")

# Patch the model with the concept vectors
# Note: the guidance scale is highly dependent on the model and concept
patch_model(model, concept_vectors, guidance_scale=32.0, guidance_layers=range(16, 32))

# Create a pipeline with the patched model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text with concept guidance
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What's it like to walk on the moon?"}], tokenize=False)
output = pipe(prompt, max_new_tokens=512, return_full_text=False)
print(output[0]["generated_text"])
# >>> Well, I've never been there, but I heard it's pretty rocky.

# Remove concept guidance
unpatch_model(model)
```

We provide the best trained concept vector found in our experiments for each model and concept in `trained_concepts/{model}/{concept}.safetensors`.
Please refer to the paper for the details on what probes and settings were used to train these vectors.

A complete example that includes streaming is given in `examples/streaming.py`.

### Training Concept Vectors

In order to train a concept vector, a corresponding dataset is required.
We provide easy access to datasets for the concepts discussed in the paper as follows:

```python
from concept_guidance.data.open_assistant import get_open_assistant_messages
from concept_guidance.data.toxic_completions import get_toxic_completions_messages
from concept_guidance.data.truthfulqa import get_truthfulqa_messages

# Humor
examples = get_open_assistant_messages(label_key="humor", max_messages=512)

# Creativity
examples = get_open_assistant_messages(label_key="creativity", max_messages=512)

# Quality
examples = get_open_assistant_messages(label_key="quality", max_messages=512)

# Compliance
# WARNING: ToxicCompletions contains offensive/harmful user prompts
examples = get_toxic_completions_messages(max_messages=512)

# Truthfulness
examples = get_truthfulqa_messages(max_messages=512)
```

It's also possible to use custom datasets.
Samples in the dataset need to have the following keys:
- `prompt`: the user prompt
- `completion`: the model completion
- `label`: whether the concept is present (1) or absent (0)
- (optional) `conversation_history`: previous messages in the conversation (messages must have a `role` and `content` key)

Example:
```json
[
  {"prompt": "How's the weather?", "completion": "It's nice and sunny outside, thanks for asking!", "label": 1},
  {"prompt": "What's it like to walk on the moon?", "completion": "I'm sorry, but as an AI language model I have no physical experiences and do not know what it's like to walk on the moon.", "label": 0},
  ...
]
```


Once the dataset is prepared, we can train concept probes as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from concept_guidance.activations import compute_activations
from concept_guidance.models.difference_in_means import DiMProbe

examples = get_examples(...)
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Compute model activations
activations, labels = compute_activations(model, tokenizer, examples)

# Train a probe on the activations
probe = DiMProbe()  # or LogisticProbe() or PCAProbe()
probe.fit(activations, labels)

# To get the vectors directly
concept_vectors = probe.get_concept_vectors()

# To save the probe
probe.save("concept.safetensors")
```


## Running the Experiments

In order to reproduce the experiments from the paper, the following steps are required.

### Setup

Clone the repository:
```bash
git clone https://github.com/dvruette/concept-guidance.git
cd concept-guidance
```

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # installs the concept_guidance package
```

### Running the scripts

There are four scripts to run the experiments:
- `scripts/train.py`: Train a probe on one of the concepts
- `scripts/generate.py`: Perform guided generation with a trained probe
- `scripts/classify.py`: Classify guided generations to measure concept adherence
- `scripts/evaluate.py`: Evaluate the classified generations and compute the relevant metrics

Example execution:
```bash
python scripts/train.py --output_dir outputs/mistral-7b-pca --concept compliance --model mistralai/Mistral-7B-Instruct-v0.1 --probe pca

# generate 19 (guidance scales) x 64 (prompts) guided samples
python scripts/generate.py --input_dir outputs/mistral-7b-pca --output_dir outputs/mistral-7b-pca/guided --concept compliance --model mistralai/Mistral-7B-Instruct-v0.1 --guidance_scale -256 -192 -128 -96 -64 -32 -16 -8 -4 0 4 8 16 32 64 96 128 192 256 --guidance_top_k 16

python scripts/classify.py --input_dir outputs/mistral-7b-pca/guided --concept compliance

python scripts/evaluate.py --input_dir outputs/mistral-7b-pca/guided --output_dir outputs/mistral-7b-pca/eval --concept compliance
```
