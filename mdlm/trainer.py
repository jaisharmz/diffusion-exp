import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from scheduler import LinearAlphaScheduler

class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "models/ModernBERT-base/tiny-shakespeare"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    num_train_epochs: float = 20
    logging_steps: float = 10
    eval_on_start: bool = False
    eval_strategy: str = "steps"
    eval_steps: float = 0.1
    save_steps: float = 0.1
    save_only_model: bool = True

class Trainer(transformers.Trainer):
    def __init__(self, *args, time_epsilon=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = LinearAlphaScheduler()
        self.time_epsilon = time_epsilon
    def _compute_loss_weights(self, t, inputs):
        b, l = inputs["input_ids"].shape
        loss_weights = -self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        return loss_weights
    def prediction_step(self, model, inputs):
        loss, outputs = self.compute_loss(model, inputs)
        logits, labels = outputs.logits.detach().contiguous(), inputs["labels"].detach().contiguous()
        return loss.detach(), logits, labels
    def compute_loss(self, model, inputs):
        input_ids, labels, attention_mask = inputs["input_ids"], inputs["labels"], inputs.get("attention_mask", None)
        b, l = input_ids.shape
        device = input_ids.device
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_indices = torch.rand((b, l), device=device) < p_mask # label = -100 ignored (not implemented)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss_weights = self._compute_loss_weights(t, input_ids)
        token_loss = F.cross_entropy(logits[masked_indices], labels[masked_indices], reduction="none") * loss_weights[masked_indices]
        loss = torch.sum(token_loss / masked_indices.sum()) / b
        return loss, outputs