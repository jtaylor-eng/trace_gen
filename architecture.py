import torch
import torch.nn as nn
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import json
from transformers import EarlyStoppingCallback
from torch.utils.data import random_split
from chess_encoder import ChessPolicyValueModel
from chess_tokenizer import create_tokenizer, process_fen

CHESS_ENC_HF_PATH = 'jrahn/ROOK-CLF-9m' 
CHESS_ENC_CKPT_PATH = './chess_enc_ckpt'
LLM_HF_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
DATASET_PATH = './datasets/dataset_USETHIS.jsonl'
MODEL_SAVE_PATH = './model_ckpts/chess_model_4task2'

class ChessLM(nn.Module):
    def __init__(
        self,
        chess_enc: str = CHESS_ENC_CKPT_PATH,
        llm_model: str = LLM_HF_PATH,
        freeze_chess_enc: bool = True,
        freeze_llm: bool = True, use_checkpoint: bool = True
    ):
        super().__init__()
        
        if use_checkpoint:
            print('using custom')
            chess_model = ChessPolicyValueModel.from_pretrained_compiled(chess_enc)
            self.c_enc = chess_model.transformer
            self.chess_dim = chess_model.config.hidden_size
        else:
            self.c_enc = AutoModelForSequenceClassification.from_pretrained(chess_enc, trust_remote_code=True).model
            self.chess_dim = 256
        
        self.c_enc = self.c_enc.float()
        for param in self.c_enc.parameters(): param.requires_grad = not freeze_chess_enc

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model, quantization_config=bnb_cfg)
        self.llm = prepare_model_for_kbit_training(self.llm)
        for param in self.llm.parameters(): param.requires_grad = not freeze_llm
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.LayerNorm(self.chess_dim),
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(torch.bfloat16)
        
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, chess_input_ids, chess_attn_mask, text_input_ids, text_attn_mask, labels=None, 
                fen_str=None, instruction=None, output=None):
        
        with torch.no_grad():
            c_ids = chess_input_ids.to(self.c_enc.device)
            c_mask = chess_attn_mask.to(self.c_enc.device)
            chess_outputs = self.c_enc(input_ids=c_ids, attention_mask=c_mask)
            chess_feats = chess_outputs.last_hidden_state
            chess_feats = torch.clamp(chess_feats, min=-1e4, max=1e4)

        chess_feats = chess_feats.to(self.projector[0].weight.dtype)
        chess_embeds = self.projector(chess_feats)

        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        full_input = torch.cat([chess_embeds, text_embeds], dim=1)
        full_mask  = torch.cat([chess_attn_mask, text_attn_mask], dim=1)
        
        combined_labels = None
        if labels is not None:
            c_ignore = torch.full((chess_embeds.shape[0], chess_embeds.shape[1]), -100, dtype=torch.long, device=labels.device)
            combined_labels = torch.cat([c_ignore, labels], dim=1)

        return self.llm(inputs_embeds=full_input, attention_mask=full_mask, labels=combined_labels)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    def get_input_embeddings(self): return self.llm.get_input_embeddings()

class ChessReasoningDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and 'fen' in obj and 'instruction' in obj and 'output' in obj:
                        self.data.append(obj)
                except: pass

    @classmethod
    def process_fen(cls, fen): return process_fen(fen)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'fen_str': self.process_fen(item['fen']), 
            'instruction': item['instruction'],
            'output': item['output']
        }
    def __len__(self): return len(self.data)

class MultimodalCollator:
    def __init__(self, chess_tokenizer, llm_tokenizer, max_text_len=1024):
        self.chess_tokenizer = chess_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_len = max_text_len
        self.llm_tokenizer.padding_side = "right"
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.separator = "\n\nAssistant:"

    def __call__(self, batch):
        if not isinstance(batch[0], dict) or 'fen_str' not in batch[0]:
            print("ERROR: Collator received invalid keys. Available keys:", batch[0].keys())
            raise KeyError("fen_str missing. Check remove_unused_columns=False or Forward signature.")

        fens = [item['fen_str'] for item in batch]
        
        c_in = self.chess_tokenizer(fens, return_tensors='pt', padding=True, truncation=True)

        text_input_ids = []
        labels_list = []

        for item in batch:
            prompt = f"User: {item['instruction']}{self.separator}"
            full_text = f"{prompt} {item['output']}{self.llm_tokenizer.eos_token}"
            
            tokenized = self.llm_tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.max_len)
            input_ids = torch.tensor(tokenized.input_ids, dtype=torch.long)
            labels = input_ids.clone()
            
            prompt_tokens = self.llm_tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_len).input_ids
            mask_len = len(prompt_tokens)
            if mask_len >= len(input_ids): mask_len = max(0, len(input_ids) - 10)
            labels[:mask_len] = -100 

            text_input_ids.append(input_ids)
            labels_list.append(labels)

        padded_inputs = torch.nn.utils.rnn.pad_sequence(text_input_ids, batch_first=True, padding_value=self.llm_tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attn_mask = (padded_inputs != self.llm_tokenizer.pad_token_id).long()

        return {
            'chess_input_ids': c_in.input_ids,
            'chess_attn_mask': c_in.attention_mask,
            'text_input_ids': padded_inputs,
            'text_attn_mask': attn_mask,
            'labels': padded_labels
        }

# def check_batch(model, collator, dataset): #dont need anymore, gradients flowing correctly
#     print("\n--- DIAGNOSTIC CHECK ---")
#     loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
#     batch = next(iter(loader))
#     batch = {k: v.to('cuda') if torch.cuda.is_available() else v for k, v in batch.items()}
    
#     lbls = batch['labels']
#     non_masked = (lbls != -100).sum().item()
#     print(f"Total Labels: {lbls.numel()}, Unmasked: {non_masked}")
    
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')
#     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#         outputs = model(**batch)
#     print(f"Computed Loss: {outputs.loss.item()}")
#     print("--- CHECK PASSED ---\n")
#     return True

def main():
    torch.cuda.empty_cache() ; gc.collect()
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)

    # Tokenizers
    tokenizer_obj = create_tokenizer()
    c_tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj, pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]")
    if c_tok.pad_token_id is None: c_tok.add_special_tokens({'pad_token': '[PAD]'})
    
    l_tok = AutoTokenizer.from_pretrained(LLM_HF_PATH)
    l_tok.padding_side = 'right'
    if l_tok.pad_token is None: l_tok.pad_token = l_tok.eos_token

    dataset = ChessReasoningDataset(DATASET_PATH)
    collator = MultimodalCollator(chess_tokenizer=c_tok, llm_tokenizer=l_tok)
    model = ChessLM()

    # if not check_batch(model, collator, dataset): return

    print('Beginning Alignment')
    alignment_args = TrainingArguments(
        output_dir="./model_ckpts/checkpoints/stage1_alignment",
        remove_unused_columns=False,
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4, 
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        gradient_checkpointing=True, 
    )

    trainer = Trainer(model=model, args=alignment_args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    torch.save(model.projector.state_dict(), os.path.join(MODEL_SAVE_PATH, 'projector_aligned.pt'))
    
    #LoRA
    print('Finished Alignment\nFine Tuning with LoRA')
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model.llm = get_peft_model(model.llm, lora_config)
    
    for param in model.projector.parameters(): param.requires_grad = True

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])

    tuning_args = TrainingArguments(
        output_dir="./model_ckpts/checkpoints/stage2_lora",
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        save_strategy="no"
    )

    trainer_stage2 = Trainer(model=model, args=tuning_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator)
    trainer_stage2.train()

    model.llm.save_pretrained(os.path.join(MODEL_SAVE_PATH, 'lora_adapters'))
    torch.save(model.projector.state_dict(), os.path.join(MODEL_SAVE_PATH, 'projector_final.pt'))
    print('Training Complete')

if __name__ == '__main__':
    main()