import torch
import torch.nn as nn
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, PreTrainedTokenizerFast, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import json
import re
from transformers import EarlyStoppingCallback
from torch.utils.data import random_split
from chess_encoder import ChessPolicyValueModel
from chess_tokenizer import create_tokenizer, process_fen

CHESS_ENC_HF_PATH = 'jrahn/ROOK-CLF-9m' 
CHESS_ENC_CKPT_PATH = './chess_enc_ckpt'
LLM_HF_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
DATASET_PATH = './datasets/dataset_no_board.jsonl'
MODEL_SAVE_PATH = './model_ckpts/chess_model_1216'

class ChessLM(nn.Module):
    def __init__(
        self,
        chess_enc: str = CHESS_ENC_CKPT_PATH,
        llm_model: str = LLM_HF_PATH,
        freeze_chess_enc: bool = True,
        freeze_llm: bool = True,
        use_checkpoint: bool = True
    ):
        super().__init__()
        if use_checkpoint:
            print('using custom')
            chess_model = ChessPolicyValueModel.from_pretrained_compiled(chess_enc)
            self.c_enc = chess_model.transformer
            self.chess_dim = chess_model.config.hidden_size
        else:
            self.c_enc = AutoModelForSequenceClassification.from_pretrained(
                chess_enc,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).model
            self.chess_dim = 256
        
        self.c_enc = self.c_enc.to(torch.bfloat16)
        for param in self.c_enc.parameters(): param.requires_grad = not freeze_chess_enc
        
        #language model
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=bnb_cfg
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        for param in self.llm.parameters(): param.requires_grad = not freeze_llm

        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(torch.bfloat16)

    def forward(self, chess_input_ids, chess_attn_mask, text_input_ids, text_attn_mask, labels=None):
        #get chess embeddings
        chess_outputs = self.c_enc(input_ids=chess_input_ids, attention_mask=chess_attn_mask)
        chess_feats = chess_outputs.last_hidden_state
        chess_feats = chess_feats.to(self.projector[0].weight.dtype)
        chess_embeds = self.projector(chess_feats)

        #get text embeddings
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        #get X,Y
        full_input = torch.cat([chess_embeds, text_embeds], dim=1)
        full_mask  = torch.cat([chess_attn_mask, text_attn_mask], dim=1)
        combined_labels = None
        if labels is not None:
            c_ignore = torch.full(
                (chess_embeds.shape[0], chess_embeds.shape[1]), 
                -100, 
                dtype=torch.long, 
                device=labels.device
            )
            combined_labels = torch.cat([c_ignore, labels], dim=1)

        return self.llm(
            inputs_embeds=full_input,
            attention_mask=full_mask,
            labels=combined_labels
        )
    
    #needed in HF trainer
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    def get_input_embeddings(self): return self.llm.get_input_embeddings()

class ChessReasoningDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []

        # with open(jsonl_path, 'r') as f:
        #     for line in f: self.data.append(json.loads(line))
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: 
                    continue # Skip empty lines
                try:
                    obj = json.loads(line)
                    # Robustness check: Ensure it's a dict and has 'fen'
                    if isinstance(obj, dict) and 'fen' in obj and 'reasoning_str' in obj:
                        self.data.append(obj)
                    else:
                        print(f"Warning: Skipping line {i} (Missing keys or not a dict)")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {i} (Invalid JSON)")

    @classmethod
    def process_fen(cls, fen):
        """Process FEN for custom tokenizer (uses process_fen from chess_tokenizer)"""
        return process_fen(fen)
    

    def __getitem__(self, idx):
        item = self.data[idx]

        fen = self.process_fen(item['fen'])
        text = item['reasoning_str']

        return {'fen_str': fen, 'text_str': text}

    def __len__(self): return len(self.data)


class MultimodalCollator:
    def __init__(self, chess_tokenizer, llm_tokenizer, max_text_len=1024):
        self.chess_tokenizer = chess_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_len = max_text_len

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

    def __call__(self, batch):
        fens = [item['fen_str'] for item in batch]
        texts = [item['text_str'] for item in batch]

        c_in = self.chess_tokenizer(
            fens,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )

        t_in = self.llm_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_len
        )

        labels = t_in.input_ids.clone()
        labels[labels==self.llm_tokenizer.pad_token_id] = -100

        return {
            'chess_input_ids': c_in.input_ids,
            'chess_attn_mask': c_in.attention_mask,
            'text_input_ids': t_in.input_ids,
            'text_attn_mask': t_in.attention_mask,
            'labels': labels
        }


def main():
    torch.cuda.empty_cache() ; gc.collect()

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    tokenizer_obj = create_tokenizer()
    c_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]"
    )
    if c_tok.pad_token_id is None:
        c_tok.add_special_tokens({'pad_token': '[PAD]'})

    
    l_tok = AutoTokenizer.from_pretrained(LLM_HF_PATH)
    if l_tok.pad_token is None: l_tok.pad_token = l_tok.eos_token

    dataset = ChessReasoningDataset(DATASET_PATH)
    collator = MultimodalCollator(chess_tokenizer=c_tok, llm_tokenizer=l_tok)

    model = ChessLM()

    print('Beginning Alignment')

    alignment_args = TrainingArguments(
        output_dir="./model_ckpts/checkpoints/stage1_alignment",
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
    print('Finished Alignment\nFine Tuning with LoRA')

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
        output_dir="./model_checkpoints/checkpoints/stage2_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
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