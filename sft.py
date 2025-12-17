import torch
import torch.nn as nn
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import json
import re
from transformers import EarlyStoppingCallback
from torch.utils.data import random_split

from architecture import (
    ChessLM,
    ChessReasoningDataset,
    MultimodalCollator,
)

CHESS_ENC_HF_PATH = 'jrahn/ROOK-CLF-9m' 
LLM_HF_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
DATASET_PATH = './dataset_no_board.jsonl'
MODEL_SAVE_PATH = './chess_model_1215'


def main():
    torch.cuda.empty_cache() ; gc.collect()

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    c_tok = AutoTokenizer.from_pretrained(CHESS_ENC_HF_PATH, trust_remote_code=True)
    l_tok = AutoTokenizer.from_pretrained(LLM_HF_PATH)
    if l_tok.pad_token is None: l_tok.pad_token = l_tok.eos_token

    dataset = ChessReasoningDataset(DATASET_PATH)
    collator = MultimodalCollator(chess_tokenizer=c_tok, llm_tokenizer=l_tok)

    model = ChessLM()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    for param in model.projector.parameters(): param.requires_grad = True
    # for param in model.c_enc.parameters(): param.requires_grad = True

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training on {train_size} samples, Validating on {val_size} samples.")

    tuning_args = TrainingArguments(
        output_dir="./checkpoints/stage2_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=1e-4,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer_stage2 = Trainer(
        model=model,
        args=tuning_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,       
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    trainer_stage2.train()

    model.llm.save_pretrained(os.path.join(MODEL_SAVE_PATH, 'lora_adapters'))
    torch.save(model.projector.state_dict(), os.path.join(MODEL_SAVE_PATH, 'projector_final.pt'))
    print('Training Complete & Models Saved')


if __name__ == '__main__':
    main()