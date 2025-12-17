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

    print('Beginning Alignment')

    alignment_args = TrainingArguments(
        output_dir="./checkpoints/stage1_alignment",
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=1e-3,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True, 
    )

    trainer = Trainer(
        model=model,
        args=alignment_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    torch.save(model.projector.state_dict(), os.path.join(MODEL_SAVE_PATH, 'projector_aligned.pt'))
    print('Finished Alignment')


if __name__ == '__main__':
    main()