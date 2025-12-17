#for checking package functionality faster than repl

import chess
from stockfish import Stockfish

# stockfish = Stockfish(path="/users/PAS3150/jacktaylor/trace_gen/Stockfish/src/stockfish")

# fen_str = 'rnbqk2r/pp4pp/2pbpn2/3p1p2/2PP4/1P3NP1/P3PPBP/RNBQ1RK1 b kq - 0 7'

# stockfish.set_fen_position(fen_str)
# board = chess.Board(fen_str)

# best_move_uci = stockfish.get_best_move()

# best_move = chess.Move.from_uci(best_move_uci)

# print(f"The best move is: {best_move}")

# top_moves = stockfish.get_top_moves(5)
# print(top_moves)

# def get_top_k_best_moves(fen: str, k: int = 5):
#     stockfish.set_fen_position(fen)
#     board = chess.Board(fen)

#     top_moves = stockfish.get_top_moves(5)

#     #top i move: centipawn score
#     moves_dict = {}
#     for entry in top_moves:
#         moves_dict[board.san(chess.Move.from_uci(entry['Move']))] = entry['Centipawn']

#     return moves_dict, f"({' '.join([f'{k} ({v})' for k,v in moves_dict.items()])})"

# print(get_top_k_best_moves(fen_str)[1])


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

tokenizer = AutoTokenizer.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)

fen = '2r2rk1/pbqnbppp/1p2pn2/2ppN3/3P4/1P1BPQ2/PBPN1PPP/3R1RK1 w - - 6 1'

def process_fen(fen):
    position, turn, castling, en_passant, halfmove, fullmove = fen.split(" ")
    position = re.sub(r'\d+', lambda m: "." * int(m.group()), position)
    position = position.replace("/", "")  # Remove row separators
    castling = castling.ljust(4, ".")     # Pad to 4 chars
    en_passant = en_passant.ljust(2, ".") # Pad to 2 chars
    halfmove = halfmove.ljust(2, ".") + "." # Pad to 3 chars total
    fullmove = fullmove.ljust(3, ".")     # Pad to 3 chars
    return "".join([position, turn, castling, en_passant, halfmove, fullmove])

fen = process_fen(fen)
fen += '[CLS]'
inputs = tokenizer(fen, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

print(out)

logits = out.logits
predicted_id = logits.argmax(-1).item()

move = model.config.id2label[predicted_id]
print("Model predicts:", move)




# def example_chess_enc_usage():
#     # Load custom model and tokenizer
#     if os.path.isdir(CHESS_ENC_CKPT_PATH) and os.path.exists(os.path.join(CHESS_ENC_CKPT_PATH, 'config.json')):
#         # Load custom checkpoint using from_pretrained_compiled to handle _orig_mod. prefixes
#         model = ChessPolicyValueModel.from_pretrained_compiled(CHESS_ENC_CKPT_PATH)
#         tokenizer_obj = create_tokenizer()
#         tokenizer = PreTrainedTokenizerFast(
#             tokenizer_object=tokenizer_obj,
#             pad_token="[PAD]",
#             unk_token="[UNK]",
#             mask_token="[MASK]"
#         )
#         if tokenizer.pad_token_id is None:
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     else:
#         tokenizer = AutoTokenizer.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
#         model = AutoModelForSequenceClassification.from_pretrained("jrahn/ROOK-CLF-9m", trust_remote_code=True)
    
#     print(model)

#     fen = '2r2rk1/pbqnbppp/1p2pn2/2ppN3/3P4/1P1BPQ2/PBPN1PPP/3R1RK1 w - - 6 1'
#     fen = ChessReasoningDataset.process_fen(fen)
#     inputs = tokenizer(fen, return_tensors='pt')

#     with torch.no_grad():
#         out = model(**inputs)

#     logits = out.logits
#     predicted_id = logits.argmax(-1).item()

#     if hasattr(model.config, 'id2label') and model.config.id2label:
#         move = model.config.id2label[predicted_id]
#         print('Model predicts: ', move)
#     else:
#         print('Model predicts move ID: ', predicted_id) 