import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import json
import re

class ChessLM(nn.Module):
    def __init__(
        self,
        chess_enc: str = 'jrahn/ROOK-CLF-9m',
        llm_model: str = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        freeze_chess_enc: bool = True,
        freeze_llm: bool = True
    ):
        #chess encoder
        self.c_enc = AutoModelForSequenceClassification.from_pretrained(chess_enc, trust_remote_code=True).model
        for param in self.c_enc.parameters(): param.requires_grad = not freeze_chess_enc
        
        #language model
        self.llm = AutoModelForCausalLM(llm_model)
        for param in self.llm.parameters(): param.requires_grad = not freeze_llm

        self.chess_dim = 256 #TODO not chess_enc drop in safe
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        )

    def forward(self, chess_input_ids, chess_attn_mask, text_input_ids, text_attn_mask, labels=None):
        #get chess embeddings
        chess_outputs = self.c_enc(input_ids = chess_input_ids, attention_mask=chess_attn_mask)
        chess_feats = chess_outputs.last_hidden_start
        chess_embeds = self.projector(chess_feats)

        #get text embeddings
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        #get X,Y
        full_input = torch.cat([chess_embeds, text_embeds], dim=1)
        full_mask  = torch.cat([chess_attn_mask, text_attn_mask], dim=1)
        combined_labels = None
        if labels:
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
    

class ChessReasoningDataset(Dataset):
    def __init__(self,):
        self.data = []
        
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
        ...

    def __len__(self): return len(self.data)


class MultimodelCollator:
    def __init__(self):
        ...

    def __call__(self, batch):
        ...


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
    example_chess_enc_usage()

    #... train

if __name__ == '__main__':
    main()