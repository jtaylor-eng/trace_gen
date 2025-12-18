import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from chess_encoder import ChessPolicyValueModel
from chess_tokenizer import create_tokenizer, process_fen
import os
import pandas as pd
import chess
import difflib

MODEL_DIR = './model_ckpts/chess_model_4task2'
CHESS_ENC_PATH = './chess_enc_ckpt' 
LLM_HF = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

class ChessLM(nn.Module):
    def __init__(self, chess_enc_path, llm_path):
        super().__init__()
        
        chess_model = ChessPolicyValueModel.from_pretrained_compiled(chess_enc_path)
        self.c_enc = chess_model.transformer
        self.chess_dim = chess_model.config.hidden_size
        print(f"  - Loaded via Custom Class. Dim: {self.chess_dim}")

        self.c_enc = self.c_enc.float()
        self.c_enc.eval() 

        print(f"Loading LLM from {llm_path}...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            quantization_config=bnb_cfg,
            device_map="auto"
        )
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.LayerNorm(self.chess_dim),
            nn.Linear(self.chess_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(torch.bfloat16)

def load_chess_model(model_dir=MODEL_DIR):
    model = ChessLM(CHESS_ENC_PATH, LLM_HF)
    
    #projector
    proj_path = os.path.join(model_dir, 'projector_final.pt')
    print(f"Loading Projector from {proj_path}...")
    state_dict = torch.load(proj_path)
    
    #ensure dimension match
    saved_dim = state_dict['0.weight'].shape[0]
    if saved_dim != model.chess_dim:
        raise ValueError(f"Dim Mismatch: Saved {saved_dim}, Model {model.chess_dim}")

    model.projector.load_state_dict(state_dict)
    
    #adapters
    adapter_path = os.path.join(model_dir, 'lora_adapters')
    print(f"Loading LoRA Adapters from {adapter_path}...")
    model.llm = PeftModel.from_pretrained(model.llm, adapter_path)
    
    return model.eval()

def generate_response(model, tokenizer, chess_tokenizer, fen, instruction, max_new_tokens=1000):
    fen_processed = process_fen(fen)
    c_in = chess_tokenizer([fen_processed], return_tensors='pt')
    
    device = model.c_enc.device 
    c_ids = c_in.input_ids.to(device)
    c_mask = c_in.attention_mask.to(device)
    
    #get chess embeddings
    with torch.inference_mode():
        c_out = model.c_enc(input_ids=c_ids, attention_mask=c_mask)
        c_feats = c_out.last_hidden_state
        c_feats = torch.nan_to_num(c_feats, nan=0.0)
        c_feats = torch.clamp(c_feats, min=-1e4, max=1e4)
        
        proj_device = model.projector[0].weight.device
        c_embeds = model.projector(c_feats.to(torch.bfloat16).to(proj_device))

    #get nl embeddings
    prompt = f"User: {instruction}\n\nAssistant:"
    llm_device = model.llm.device 
    t_in = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).to(llm_device)
    
    with torch.inference_mode():
        t_embeds = model.llm.get_input_embeddings()(t_in.input_ids)

    c_embeds = c_embeds.to(llm_device)
    inputs_embeds = torch.cat([c_embeds, t_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=llm_device)
    

    with torch.inference_mode(): 
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# def validate_board_extraction(model, llm_tok, c_tok_wrapped, fens):
#     instruction = 'Extract the board representation from this chess position.'
#     em_count = 0
#     for fen in fens:
#         response = generate_response(
#             model, 
#             llm_tok, 
#             c_tok_wrapped, 
#             fen, 
#             instruction
#         )

#         if response == str(chess.Board(fen)):
#             em_count += 1
        

# def validate_fen_extraction(model, fens):
#     ...

def validate_board_extraction(model, llm_tok, c_tok_wrapped, fens):
    instruction = 'Extract the board representation from this chess position.'
    em_count = 0
    total_similarity = 0.0
    
    print(f"\n--- Validating Board Extraction ({len(fens)} samples) ---")
    
    for i, fen in enumerate(fens):
        response = generate_response(model, llm_tok, c_tok_wrapped, fen, instruction, max_new_tokens=256)
        
        gt_board_str = str(chess.Board(fen))
        
        pred_clean = response.strip()
        gt_clean = gt_board_str.strip()

        if pred_clean == gt_clean: em_count += 1

        sim = difflib.SequenceMatcher(None, pred_clean, gt_clean).ratio()
        total_similarity += sim
        
        if (i+1) % 10 == 0: 
            print(f"  Processed {i+1}/{len(fens)} | Current Avg Sim: {total_similarity/(i+1):.4f}")

    print(f"\n[Board Extraction Results]")
    print(f"  Exact Match:   {em_count}/{len(fens)} ({100*em_count/len(fens):.2f}%)")
    print(f"  Avg Similarity: {total_similarity/len(fens):.4f}")


def validate_fen_extraction(model, llm_tok, c_tok_wrapped, fens):
    instruction = 'Extract the FEN notation from this chess position.'
    em_count = 0
    total_piece_accuracy = 0.0
    valid_syntax_count = 0
    
    print(f"\n--- Validating FEN Extraction ({len(fens)} samples) ---")
    
    for i, fen in enumerate(fens):
        response = generate_response(model, llm_tok, c_tok_wrapped, fen, instruction, max_new_tokens=128)
        
        pred_fen = response.strip()
        gt_fen = fen.strip()

        if pred_fen == gt_fen: em_count += 1

        gt_board = chess.Board(gt_fen)
        try:
            pred_board = chess.Board(pred_fen)
            valid_syntax_count += 1
            
            matches = 0
            for square in chess.SQUARES:
                if gt_board.piece_at(square) == pred_board.piece_at(square):
                    matches += 1
            
            accuracy = matches / 64.0
            total_piece_accuracy += accuracy
            
        except ValueError: pass #invalid board
            
        if (i+1) % 10 == 0: 
            print(f"  Processed {i+1}/{len(fens)}")

    print(f"\n[FEN Extraction Results]")
    print(f"  Exact Match:    {em_count}/{len(fens)} ({100*em_count/len(fens):.2f}%)")
    print(f"  Valid Syntax:   {valid_syntax_count}/{len(fens)}")
    print(f"  Piece Accuracy: {total_piece_accuracy/len(fens):.4f} (Avg % of board correct)")

if __name__ == '__main__':
    #tokenizers
    chess_tok = create_tokenizer()
    from transformers import PreTrainedTokenizerFast
    c_tok_wrapped = PreTrainedTokenizerFast(
        tokenizer_object=chess_tok,
        pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]"
    )
    if c_tok_wrapped.pad_token_id is None: c_tok_wrapped.add_special_tokens({'pad_token': '[PAD]'})

    llm_tok = AutoTokenizer.from_pretrained(LLM_HF)
    
    model = load_chess_model()
    
    df = pd.read_csv('./datasets/lichess_puzzles_val.csv')
    val_fens = list(df['FEN'])

    # test_cases = [
    #     {
    #         "fen": "r1b1kb1r/pp3ppp/1qp5/3pPn2/3P4/2N1B3/P3BPPP/2RQ1RK1 w kq - 3 13", 
    #         "instruction": "Analyze this chess position and find the best move."
    #     },
    #     {
    #         "fen": "r1b1kb1r/pp3ppp/1qp5/3pPn2/3P4/2N1B3/P3BPPP/2RQ1RK1 w kq - 3 13",
    #         "instruction": "Extract the FEN notation from this chess position."
    #     },
    #     {
    #         "fen": "r1b1kb1r/pp3ppp/1qp5/3pPn2/3P4/2N1B3/P3BPPP/2RQ1RK1 w kq - 3 13",
    #         "instruction": "Extract the board representation from this chess position."
    #     }
    # ]

    # print("\n" + "="*50)
    # for i, case in enumerate(test_cases):
    #     print(f"\nTest Case {i+1}: {case['instruction']}")
    #     print(f"FEN: {case['fen']}")
        
    #     response = generate_response(
    #         model, 
    #         llm_tok, 
    #         c_tok_wrapped, 
    #         case['fen'], 
    #         case['instruction']
    #     )
        
    #     print(f"\nResponse:\n{response}")
    #     print("-" * 50)
    
    validate_board_extraction(model, llm_tok, c_tok_wrapped, val_fens)
    validate_fen_extraction(model, llm_tok, c_tok_wrapped, val_fens)