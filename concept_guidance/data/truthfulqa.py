from datasets import load_dataset
from truthfulqa.presets import preset_map


def get_truthfulqa_messages(dataset_name="truthful_qa", subcorpus="generation", max_messages=-1, do_few_shot=False, cache_dir=None):
    conversation_history = None
    if do_few_shot:
        # get the few shot prompt
        sentences = preset_map["qa"].split('\n')
        # remove the "Q: " and "A: " prefixes
        questions, answers = [x[3:] for x in sentences[::3]], [x[3:] for x in sentences[1::3]]

        conversation_history = []
        for question, answer in zip(questions, answers):
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

    ds = load_dataset(dataset_name, subcorpus, cache_dir=cache_dir)
    ds_train = ds["validation"].shuffle(seed=0)
    messages = []
    for x in ds_train:
        if max_messages > 0 and len(messages) >= max_messages:
            break

        if len(x["correct_answers"]) == 0 or len(x["incorrect_answers"]) == 0:
            continue
        
        messages.append({
            "prompt": x["question"],
            "completion": x["best_answer"],
            "label": 1,
            "conversation_history": conversation_history
        })
        for answer in x["incorrect_answers"][:1]:
            messages.append({
                "prompt": x["question"],
                "completion": answer,
                "label": 0,
                "conversation_history": conversation_history
            })
    return messages[:max_messages]
