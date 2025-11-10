from transformers import AutoTokenizer, AutoModelForMaskedLM

def get_model_and_tokenizer(model_name="answerdotai/ModernBERT-large"):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer