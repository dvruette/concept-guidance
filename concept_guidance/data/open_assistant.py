import random
from typing import Literal
from datasets import load_dataset


OALabelKey = Literal[
    "spam", "fails_task", "pii", "not_approproate", "hate_speech", "sexual_content", "quality", "toxicity", "humor", "helpfulness", "creativity", "violence"
]

def get_open_assistant_messages(
    dataset_name: str = "OpenAssistant/oasst1",
    lang: str = "en",
    label_key: OALabelKey = "quality",
    max_messages: int = 1000,
    seed: int = 0,
    do_few_shot=False,
    cache_dir=None
):
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    ds = ds.filter(lambda x: x["lang"] == lang)
    ds_train = ds["train"].to_list()
    roots = {x["message_id"]: x for x in ds_train if x["parent_id"] is None}
    messages = [(roots[x["parent_id"]], x) for x in ds_train if x["parent_id"] in roots]

    def keep_message(msg_pair):
        _, assistant_msg = msg_pair
        if assistant_msg.get("deleted", False):
            return False
        if not "labels" in assistant_msg or assistant_msg["labels"] is None:
            return False
        if label_key not in assistant_msg.get("labels", {}).get("name", []):
            return False
        return True
    filtered_messages = list(filter(keep_message, messages))

    sorted_messages = sorted(filtered_messages, key=lambda x: x[1]["labels"]["value"][x[1]["labels"]["name"].index(label_key)])
    if len(sorted_messages) < max_messages:
        print(f"WARNING: Found {len(sorted_messages)} messages, but requested {max_messages}")
        max_messages = len(sorted_messages)
    num_bad = max_messages // 2
    num_good = max_messages - num_bad
    
    padding = 3 if do_few_shot else 0
    completion_history = None
    if do_few_shot:
        completion_history = []
        for i in range(3):
            completion_history.append({"role": "user", "content": sorted_messages[i][0]["text"]})
            completion_history.append({"role": "assistant", "content": sorted_messages[i][1]["text"]})
            completion_history.append({"role": "user", "content": sorted_messages[-i][0]["text"]})
            completion_history.append({"role": "assistant", "content": sorted_messages[-i][1]["text"]})

    
    worst_messages = sorted_messages[padding:num_bad + padding]
    best_messages = sorted_messages[-num_good - padding : len(sorted_messages) - padding]
    if len(best_messages) != len(worst_messages):
        print(f"WARNING: Found {len(best_messages)} best messages and {len(worst_messages)} worst messages")
    
    labelled_messages = (
        [
            {
                "prompt": user_msg["text"],
                "completion": assistant_msg["text"],
                "conversation_history": completion_history,
                "label": 1,
            }
            for user_msg, assistant_msg in best_messages
        ] + [
            {
                "prompt": user_msg["text"],
                "completion": assistant_msg["text"],
                "conversation_history": completion_history,
                "label": 0,
            }
            for user_msg, assistant_msg in worst_messages
        ]
    )
    random.seed(seed)
    random.shuffle(labelled_messages)
    return labelled_messages
