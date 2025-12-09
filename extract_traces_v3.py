import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import chess
import chess.pgn
import re
import json
import time
from stockfish import Stockfish

#Constants / config
STOCKFISH = Stockfish(path="/users/PAS3150/jacktaylor/trace_gen/Stockfish/src/stockfish")

MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct' #'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
BATCH_SIZE = 16
MAX_NEW_TOKENS = 1024 
PGN_FILE = 'game_studies_new.pgn'
OUTPUT_FILE = 'train_chess_reasoning_gptoss2.jsonl'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "left"
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
MODEL.config.use_cache = True

batch_buffer = []


def clean_comment(comment):
    comment = re.sub(r'\[%[^\]]+\]', '', comment) #remove [%...]
    comment = re.sub(r'\s+', ' ', comment).strip() #remove excess whitespace
    return comment


def construct_prompt(fen, history_str, best_moves, move_san, raw_comment):
    """
    Construct a concise prompt for the chess reasoning model.
    Focus on candidate moves, reasoning, and final move.
    Keep it short to avoid token overflow.
    """
    return f"""
You are an expert chess analyst. Given a chess position in FEN notation a list of candidate moves, and some human commentary on the chosen best move, generate a JSON output with the following structure:
{{
  "candidates": [
    {{
      "move": "<candidate move>",
      "pros": "<reason why this move could be good>",
      "cons": "<potential drawbacks or risks>"
    }},
    ...
  ],
  "chosen_move_reasoning": "<a detailed explanation of why the chosen move is the best in this position>",
}}

Input Format:
FEN: FEN string of chess position.
Candidate moves: List of moves in SAN format followed by Centipawn score (positive better for white, negative better for black).
Best move: Move string in SAN format.
Best move commentary: A natural language string to help with your justification.

Requirements:
- The JSON must be valid and properly formatted.
- Include at least 3 candidate moves.
- Be concise but precise in pros, cons, and reasoning.
- The 'original_comment' should read like something a chess commentator would say, in natural language.
- Only return the JSON, nothing else.

Input example:
FEN: "rnbqk2r/pp4pp/2pbpn2/3p1p2/2PP4/1P3NP1/P3PPBP/RNBQ1RK1 b kq - 0 7"
Candidate moves: (O-O (25) Qe7 (25) Ne4 (29) b6 (34) Nbd7 (35)) 
Best move: "Qe7"
Best move commentary:  Correct! In this pawn structure (a so-called Stonewall), with most of our pawns on light squares (f5, e6, d5, c6), our dark-squared bishop is an active piece (as well as an imporatnt defender of weakened dark squares) and we should not allow white to play Ba3 (with the bishops exachange) under positive circumstances!

Output example:
{{
  "candidates": [
    {{
      "SAN": "Qe7",
      "pros": [
        "Connects rooks",
        "Prepares kingside castling",
        "Supports central pawn on e6"
      ],
      "cons": [
        "Passive move, does not create immediate threats"
      ]
    }},
    {{
      "SAN": "O-O",
      "pros": [
        "King safety",
        "Rook activation"
      ],
      "cons": [
        "Leaves queen undeveloped",
        "No immediate pressure on center"
      ]
    }},
    {{
      "SAN": "Ne4",
      "pros": [
        "Centralizes knight",
        "Attacks c3 pawn",
        "Potential tactical ideas"
      ],
      "cons": [
        "May be captured by f3 knight",
        "Exposes knight to attacks"
      ]
    }},
    {{
      "SAN": "b6",
      "pros": [
        "Prepares Bb7 development",
        "Supports c5 break"
      ],
      "cons": [
        "Slower, does not develop king or rook"
      ]
    }},
    {{
      "SAN": "Nbd7",
      "pros": [
        "Connects rooks later",
        "Supports central control"
      ],
      "cons": [
        "Less active, blocks c8 bishop temporarily"
      ]
    }}
  ],
  "chosen_move_reasoning": "Qe7 is chosen because it develops the queen to a safe and active square, connects the rooks, supports central pawns, and prepares for kingside castling. While not immediately aggressive, it follows sound opening principles of development, king safety preparation, and central control.",
}}

Output ONLY the valid JSON and NOTHING else! Carefully reread all instructions before continuing.

Input:
FEN: {fen}
Candidate Moves: {best_moves}
Best Move: {move_san}
Best Move Commentary: {raw_comment}

Output:
"""


def process_batch(f):
    global batch_buffer
    if not batch_buffer: 
        return

    prompts = [item['prompt'] for item in batch_buffer]

    inputs = TOKENIZER(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(MODEL.device)

    with torch.no_grad():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            early_stopping=True,
            pad_token_id=TOKENIZER.eos_token_id
        )

    prompt_lengths = [len(inputs.input_ids[i]) for i in range(len(prompts))]
    
    for i, output_ids in enumerate(generated_ids):
        new_tokens = output_ids[prompt_lengths[i]:]
        text = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean garbage tokens
        text = re.sub(r"[?\u2026]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 10:
            entry = {
                "fen": batch_buffer[i]['fen'],
                "move": batch_buffer[i]['move'],
                "reasoning_trace": text,
                "original_comment": batch_buffer[i]['raw_comment']
            }
            f.write(json.dumps(entry) + '\n')
        else:
            print("REJECTED: reasoning too short or invalid")

    f.flush()
    batch_buffer = []

    # print('finished trial run') ; exit()


def get_prev_k_moves(board: chess.Board, k: int = 3) -> str:
    move_history = []
    try:
        if len(board.move_stack) > 0:
            replay_board = chess.Board() 
            for m in board.move_stack[k:]: #last k moves
                move_history.append(replay_board.san(m))
                replay_board.push(m)

        history_str = ' '.join(move_history)
    except: history_str = 'N/A'

    return f"history str: {history_str}"


def get_top_k_best_moves(fen: str, k: int = 5):
    STOCKFISH.set_fen_position(fen)
    board = chess.Board(fen)

    top_moves = STOCKFISH.get_top_moves(5)

    #top i move: centipawn score
    moves_dict = {}
    for entry in top_moves:
        moves_dict[board.san(chess.Move.from_uci(entry['Move']))] = entry['Centipawn']

    return moves_dict, f"({' '.join([f'{k} ({v})' for k,v in moves_dict.items()])})"


def process_node(node, board, f_out):
    """Recursively walk PGN tree"""
    raw_comment = node.comment
    cleaned_comment = clean_comment(raw_comment)
    
    if cleaned_comment and len(cleaned_comment) > 10:
        fen = board.fen()
        move_san = node.san()
        
        best_move_dict, best_move_str = get_top_k_best_moves(fen)
        if move_san not in best_move_dict: return #suboptimal lichess study suggestion

        #add in buffer
        batch_buffer.append({
            "prompt": construct_prompt(
                fen,
                get_prev_k_moves(board),
                best_move_str,
                move_san,
                cleaned_comment
            ),
            "fen": fen,
            "move": move_san,
            "raw_comment": cleaned_comment
        })

        #batch full, begin processing
        if len(batch_buffer) >= BATCH_SIZE: process_batch(f_out)


    if node.move in board.legal_moves: board.push(node.move)
    else: return #illegal move, etc

    #main line recursion
    if node.next():
        process_node(node.next(), board, f_out)
    
    #variations recursion (may be included in study)
    for variation in node.variations:
        board.pop() #undo main move
        
        #check legality
        if variation.move in board.legal_moves:
            board.push(variation.move)
            if variation.next():
                process_node(variation.next(), board, f_out)
            board.pop()
        
        board.push(node.move) #restore main move

    board.pop()

def main():
    print(f'Starting pipeline. Batch Size: {BATCH_SIZE}')
    start_time = time.time()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with open(PGN_FILE, 'r', encoding='utf-8') as pgn_file:
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None: break #finished with file
                
                game_count += 1
                if game_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f'Parsed Game #{game_count} | Time elapsed: {elapsed:.2f}s')

                board = game.board()
                
                if game.next(): 
                    process_node(game.next(), board, f_out)

if __name__ == "__main__":
    main()