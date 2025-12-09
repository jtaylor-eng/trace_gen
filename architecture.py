import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import re

class ChessLM(nn.Module):
    def __init__(
        self,
        chess_enc,
        llm_model,
        freeze_chess_enc: bool = True,
        freeze_llm: bool = True
    ):
        self.c_enc = chess_enc
        for param in self.c_enc.parameters(): param.requires_grad = not freeze_chess_enc

        self.llm = AutoModelForCausalLM(llm_model)
        for param in self.llm.parameters(): param.requires_grad = not freeze_llm

        self.chess_dim = ... #TODO: find
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        )

    def forward(self, c_in, t_in, labels=None):
        c_feats = self.c_enc(c_in)
        c_embds = self.projector(c_feats)

        if labels:
            ...
        else:
            ...

        return self.llm(...)
    

def process_fen(fen):
    position, turn, castling, en_passant, halfmove, fullmove = fen.split(" ")
    position = re.sub(r'\d+', lambda m: "." * int(m.group()), position)
    position = position.replace("/", "")  # Remove row separators
    castling = castling.ljust(4, ".")     # Pad to 4 chars
    en_passant = en_passant.ljust(2, ".") # Pad to 2 chars
    halfmove = halfmove.ljust(2, ".") + "." # Pad to 3 chars total
    fullmove = fullmove.ljust(3, ".")     # Pad to 3 chars
    return "".join([position, turn, castling, en_passant, halfmove, fullmove])


def main():
    tokenizer = AutoTokenizer.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
    print(model)

    fen = '2r2rk1/pbqnbppp/1p2pn2/2ppN3/3P4/1P1BPQ2/PBPN1PPP/3R1RK1 w - - 6 1'
    fen = process_fen(fen)
    fen += '[CLS]'
    inputs = tokenizer(fen, return_tensors='pt')

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits
    predicted_id = logits.argmax(-1).item()

    move = model.config.id2label[predicted_id]
    print('Model predicts: ', move) 

if __name__ == '__main__':
    main()