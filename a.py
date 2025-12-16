
# gemini vibe coded to quickly vibe check model
# may refine later

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import chess
import re
import os

CHESS_ENC_PATH = 'jrahn/ROOK-CLF-9m' 
LLM_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
PROJECTOR_PATH = "./chess_model_1215/projector_final.pt"
LORA_PATH = "./chess_model_1215/lora_adapters"

class ChessLM(nn.Module):
    def __init__(self):
        super().__init__()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_PATH, 
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        self.device = self.llm.device
        self.c_enc = AutoModelForSequenceClassification.from_pretrained(
            CHESS_ENC_PATH, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 
        ).model.to(self.device)
        
        self.chess_dim = 256
        self.llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(dtype=torch.bfloat16, device=self.device)

    def get_chess_embeddings(self, input_ids, attn_mask):
        with torch.no_grad():
            outputs = self.c_enc(input_ids=input_ids, attention_mask=attn_mask)
            feats = outputs.last_hidden_state
            # Ensure feats match projector dtype
            feats = feats.to(self.projector[0].weight.dtype)
            return self.projector(feats)

# --- Utils ---
def process_fen_for_encoder(fen):
    """Standardizes FEN format for the jrahn encoder"""
    position, turn, castling, en_passant, halfmove, fullmove = fen.split(" ")
    position = re.sub(r'\d+', lambda m: "." * int(m.group()), position)
    position = position.replace("/", "")  
    castling = castling.ljust(4, ".")     
    en_passant = en_passant.ljust(2, ".") 
    halfmove = halfmove.ljust(2, ".") + "." 
    fullmove = fullmove.ljust(3, ".")     
    out = "".join([position, turn, castling, en_passant, halfmove, fullmove])
    return out + '[CLS]'

def get_text_board(fen):
    board = chess.Board(fen)
    return f"{fen}\n{board}"

# --- Inference ---
def run_inference(fen, model, c_tok, l_tok, max_new_tokens=512):
    device = model.device # Use the class attribute we set

    # 1. Prepare "Vision" (Chess Encoder Input)
    processed_fen = process_fen_for_encoder(fen)
    c_inputs = c_tok(processed_fen, return_tensors='pt').to(device)
    chess_embeds = model.get_chess_embeddings(c_inputs.input_ids, c_inputs.attention_mask)

    # 2. Prepare Text Prompt
    board_text = get_text_board(fen)
    print(board_text)
    # Note: Added \n after <think> and before board to match your training example
    prompt = f"User: Analyze this chess position and find the best move.\n\nAssistant: \n<think>"
    
    t_inputs = l_tok(prompt, return_tensors='pt').to(device)
    text_embeds = model.llm.get_input_embeddings()(t_inputs.input_ids)

    # 3. Concatenate
    inputs_embeds = torch.cat([chess_embeds, text_embeds], dim=1)
    
    # 4. Attention Mask
    chess_mask = torch.ones(chess_embeds.shape[:2], device=device, dtype=torch.long)
    combined_mask = torch.cat([chess_mask, t_inputs.attention_mask], dim=1)

    print(f"Thinking about:\n{prompt}")

    # 5. Generate
    with torch.no_grad():
        outputs = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=l_tok.eos_token_id
        )

    return l_tok.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Loading Tokenizers...")
    c_tok = AutoTokenizer.from_pretrained(CHESS_ENC_PATH, trust_remote_code=True)
    l_tok = AutoTokenizer.from_pretrained(LLM_PATH)

    print("Loading Base Model...")
    model = ChessLM()

    print("Loading Weights...")
    if os.path.exists(PROJECTOR_PATH):
        model.projector.load_state_dict(torch.load(PROJECTOR_PATH))
    else:
        print(f"ERROR: {PROJECTOR_PATH} not found.")
        return

    model.llm = PeftModel.from_pretrained(model.llm, LORA_PATH)
    model.eval()
    
    print("\n" + "="*50)
    print("Chess Reasoner Ready! Enter a FEN string.")
    print("="*50 + "\n")

    while True:
        fen = input("FEN (or 'q'): ").strip()
        if fen.lower() == 'q': break
        if not fen: continue
        
        try:
            response = run_inference(fen, model, c_tok, l_tok)
            print("\n--- Response ---\n")
            print(response)
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()