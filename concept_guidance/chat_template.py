
_chat_template = """
{{ bos_token }}
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ 'Question: ' }}
    {% elif message['role'] == 'assistant' %}
        {{ 'Answer: ' }}
    {% endif %}
    {{ message['content'].strip() }}
    {{ '\\n' }}
{% endfor %}
"""
DEFAULT_CHAT_TEMPLATE = "".join(line.strip() for line in _chat_template.split("\n"))
