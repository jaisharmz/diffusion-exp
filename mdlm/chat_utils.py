import torch

class GeneratorConfig:
    max_new_tokens: int = 128
    max_length: int = None
    block_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None

# def decode_trim(tokenizer, seq_ids_list, input_ids_list):
#     sequences = []
#     for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
#         full = list(seq_ids)
#         prompt = list(input_ids)
#         pad_id = tokenizer.pad_token_id
#         while full and full[0] == pad_id:
#             full.pop(0)
#         start = len(prompt)
#         end = len(full)
#         eos_id = tokenizer.eos_token_id
#         eot_id = tokenizer.eot_token_id
#         for i in range(start, len(full)):
#             if full[i] in (eos_id, eot_id):
#                 end = i
#                 break
        
#         gen_ids = full[start:end]
#         text = tokenizer.decode(gen_ids, skip_special_tokens=False)
#         eos = tokenizer.eos_token
#         eot = tokenizer.eot_token
#         text = text.split(eos)[0]
#         text = text.split(eot)[0]
#         sequences.append(text)
#     return sequences

def decode_trim(tokenizer, seq_ids_list, input_ids_list):
    sequences = []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    eot_id = tokenizer.eot_token_id
    
    # Handle cases where some are None
    stop_ids = {pad_id, eos_id, eot_id} - {None}

    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        prompt_len = len(input_ids)
        
        # Remove any leading padding
        while prompt_len > 0 and input_ids[0] == pad_id:
            input_ids = input_ids[1:]
            prompt_len -= 1

        # Find the start of the generated text
        start = prompt_len
        
        # Find the end of the generated text
        full_ids = list(seq_ids)
        end = len(full_ids)
        
        for i in range(start, len(full_ids)):
            if full_ids[i] in stop_ids:
                end = i
                break
        
        gen_ids = full_ids[start:end]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        # Manual extra clean for safety
        if tokenizer.eos_token:
            text = text.split(tokenizer.eos_token)[0]
        if tokenizer.eot_token:
            text = text.split(tokenizer.eot_token)[0]
            
        sequences.append(text.strip())
    return sequences

def single_turn_generate(generator):
    gen_config = GeneratorConfig()
    model, tokenizer = generator.model, generator.tokenizer
    user_text = input("Prompt here: ").strip()
    messages = [{"role": "user", "content": user_text}]
    inputs_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer([inputs_text], add_special_tokens=False)["input_ids"]
    outputs = generator.generate(inputs, gen_config, return_dict_in_generate=True)
    text = decode_trim(tokenizer, outputs["sequences"].tolist(), inputs)[0]
    print(text)