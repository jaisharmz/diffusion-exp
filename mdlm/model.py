from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

def get_model_and_tokenizer(model_name="answerdotai/ModernBERT-large"):
    config = AutoConfig.from_pretrained(model_name)
    config.reference_compile = False
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = ["[SYS]", "[/SYS]", "[Question]", "[/Question]", "[Answer]", "[/Answer]"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.eot_token = "[/Answer]"
    tokenizer.eos_token = "[/Answer]"
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    return model, tokenizer