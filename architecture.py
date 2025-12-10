import torch
import torch.nn as nn
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import json
import re

CHESS_ENC_HF_PATH = 'jrahn/ROOK-CLF-9m' 
LLM_HF_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
DATASET_PATH = './dataset.jsonl'
MODEL_SAVE_PATH = './final_chess_model/.'

class ChessLM(nn.Module):
    def __init__(
        self,
        chess_enc: str = CHESS_ENC_HF_PATH,
        llm_model: str = LLM_HF_PATH,
        freeze_chess_enc: bool = True,
        freeze_llm: bool = True
    ):
        super().__init__()
        #chess encoder
        self.c_enc = AutoModelForSequenceClassification.from_pretrained(
            chess_enc,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).model
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

        self.chess_dim = 256 #TODO not chess_enc drop in safe
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
        """https://huggingface.co/jrahn/ROOK-CLF-9m"""
        position, turn, castling, en_passant, halfmove, fullmove = fen.split(" ")
        position = re.sub(r'\d+', lambda m: "." * int(m.group()), position)
        position = position.replace("/", "")  # Remove row separators
        castling = castling.ljust(4, ".")     # Pad to 4 chars
        en_passant = en_passant.ljust(2, ".") # Pad to 2 chars
        halfmove = halfmove.ljust(2, ".") + "." # Pad to 3 chars total
        fullmove = fullmove.ljust(3, ".")     # Pad to 3 chars
        out = "".join([position, turn, castling, en_passant, halfmove, fullmove])
        return out + '[CLS]'
    

    def __getitem__(self, idx):
        item = self.data[idx]

        fen = self.process_fen(item['fen'])
        text = item['reasoning_str']

        return {'fen_str': fen, 'text_str': text}

    def __len__(self): return len(self.data)


class MultimodelCollator:
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


def example_chess_enc_usage():
    tokenizer = AutoTokenizer.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
    print(model)

    fen = '2r2rk1/pbqnbppp/1p2pn2/2ppN3/3P4/1P1BPQ2/PBPN1PPP/3R1RK1 w - - 6 1'
    fen = ChessReasoningDataset.process_fen(fen)
    inputs = tokenizer(fen, return_tensors='pt')

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits
    predicted_id = logits.argmax(-1).item()

    move = model.config.id2label[predicted_id]
    print('Model predicts: ', move) 


def main():
    # example_chess_enc_usage()

    torch.cuda.empty_cache() ; gc.collect()

    c_tok = AutoTokenizer.from_pretrained(CHESS_ENC_HF_PATH, trust_remote_code=True)
    l_tok = AutoTokenizer.from_pretrained(LLM_HF_PATH)
    if l_tok.pad_token is None: l_tok.pad_token = l_tok.eos_token

    dataset = ChessReasoningDataset(DATASET_PATH)
    collator= MultimodelCollator(chess_tokenizer=c_tok, llm_tokenizer=l_tok)

    model = ChessLM()

    #Alignment (freeze chess_enc, llm, train projection matrix only)
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

    torch.save(mode.projector.state_dict(), MODEL_SAVE_PATH + 'projector_aligned.pt')
    print('Finished Alignment')

    #LoRA tuning
    print('Fine Tuning with LoRA')

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameter()

    for param in model.projector.parameters():
        param.requires_grad = True

        tuning_args = TrainingArguments(
        output_dir="./checkpoints/stage2_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )

    trainer_stage2 = Trainer(
        model=model,
        args=tuning_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer_stage2.train()

    model.llm.save_pretrained(MODEL_SAVE_PATH + 'lora_adapters')
    torch.save(model.projector.state_dict(), MODEL_SAVE_PATH + 'projector_final.pt')
    print('Training Complete & Models Saved')

if __name__ == '__main__':
    main()