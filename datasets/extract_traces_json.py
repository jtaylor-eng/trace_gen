import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import chess
import chess.pgn
import re
import json
import time
from stockfish import Stockfish

MIN_LEN_COMMENT = 32 #do not use process study if len(commentary) less than

STOCKFISH = Stockfish(path="/users/PAS3150/jacktaylor/trace_gen/Stockfish/src/stockfish")

MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
BATCH_SIZE = 32
MAX_NEW_TOKENS = 1024
PGN_FILE = 'game_studies_new.pgn'
OUTPUT_FILE = 'train_chess_reasoning.jsonl'

SYSTEM_PROMPT = """You are an expert chess analyst. Given a chess position in FEN notation, a list of candidate moves, and some human commentary on the chosen best move, generate a JSON output.

Your output must follow this exact structure:
{
  "candidates": [
    {
      "SAN": "<candidate move in SAN>",
      "pros": ["<reason 1>", "<reason 2>"],
      "cons": ["<risk 1>"]
    }
  ],
  "chosen_move_reasoning": "<detailed explanation of why the chosen move is best>"
}

Requirements:
1. Return ONLY valid JSON.
2. Include at least 3 candidate moves in your analysis.
3. The 'chosen_move_reasoning' should assimilate the provided human commentary but sound professional.
"""

ONE_SHOT_INPUT = """
FEN: rnbqk2r/pp4pp/2pbpn2/3p1p2/2PP4/1P3NP1/P3PPBP/RNBQ1RK1 b kq - 0 7
Candidate Moves: O-O (25) Qe7 (25) Ne4 (29) b6 (34) Nbd7 (35)
Best Move: Qe7
Best Move Commentary: Correct! In this pawn structure (a so-called Stonewall), with most of our pawns on light squares (f5, e6, d5, c6), our dark-squared bishop is an active piece and we should not allow white to play Ba3!
"""

ONE_SHOT_OUTPUT = """
{
  "candidates": [
    {
      "SAN": "Qe7",
      "pros": [
        "Connects rooks",
        "Prepares kingside castling",
        "Prevents White's Ba3 exchange plan"
      ],
      "cons": [
        "Passive compared to immediate tactical strikes"
      ]
    },
    {
      "SAN": "O-O",
      "pros": [
        "King safety",
        "Rook activation"
      ],
      "cons": [
        "Leaves queen undeveloped",
        "Allows White's Ba3 idea"
      ]
    },
    {
      "SAN": "Ne4",
      "pros": [
        "Centralizes knight",
        "Attacks c3"
      ],
      "cons": [
        "Premature, can be kicked or exchanged"
      ]
    }
  ],
  "chosen_move_reasoning": "Qe7 is the best choice. In this Stonewall structure, Black's dark-squared bishop is a crucial asset. Qe7 proactively prevents White from playing Ba3 to exchange bishops. It also develops the queen and connects the rooks."
}
"""

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
    comment = re.sub(r'\[%[^\]]+\]', '', comment) # remove [%...]
    comment = re.sub(r'\s+', ' ', comment).strip() # remove excess whitespace
    return comment

def construct_messages(fen, history_str, best_moves, move_san, raw_comment):
    """
    Constructs the list of messages for the chat template.
    Uses System -> User (Example) -> Assistant (Example) -> User (Actual).
    """
    
    # Format the current input to match the one-shot format
    current_input_text = f"""
FEN: {fen}
History: {history_str}
Candidate Moves: {best_moves}
Best Move: {move_san}
Best Move Commentary: {raw_comment}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ONE_SHOT_INPUT.strip()},
        {"role": "assistant", "content": ONE_SHOT_OUTPUT.strip()},
        {"role": "user", "content": current_input_text.strip()}
    ]
    return messages

def process_batch(f):
    global batch_buffer
    if not batch_buffer: 
        return

    # Apply Chat Template
    # Llama 3 handles chat templates by adding specific header tokens
    prompts = [
        TOKENIZER.apply_chat_template(
            item['messages'], 
            tokenize=False, 
            add_generation_prompt=True
        ) for item in batch_buffer
    ]

    inputs = TOKENIZER(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_NEW_TOKENS + 512 # Allow room for prompt + generation
    ).to(MODEL.device)

    with torch.no_grad():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # Deterministic greedy decoding usually better for JSON structure
            early_stopping=True,
            pad_token_id=TOKENIZER.eos_token_id
        )

    prompt_lengths = [len(inputs.input_ids[i]) for i in range(len(prompts))]
    
    for i, output_ids in enumerate(generated_ids):
        new_tokens = output_ids[prompt_lengths[i]:]
        text = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

        # Basic cleanup of markdown code blocks if the model adds them
        text = text.replace("```json", "").replace("```", "").strip()

        if len(text) > MIN_LEN_COMMENT:
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

def get_prev_k_moves(board: chess.Board, k: int = 3) -> str:
    """
    Replays the game to get the SAN notation of the LAST k moves.
    """
    if not board.move_stack:
        return "N/A"

    try:
        # Create a fresh board to replay moves to get correct SAN context
        replay_board = chess.Board()
        san_history = []
        
        # We need to replay up to the current state to generate SAN correctly
        for move in board.move_stack:
            san_history.append(replay_board.san(move))
            replay_board.push(move)
        
        # Slice the last k moves
        last_k = san_history[-k:] if len(san_history) >= k else san_history
        return ' '.join(last_k)
    except Exception as e:
        return 'N/A'

def get_top_k_best_moves(fen: str, k: int = 5):
    STOCKFISH.set_fen_position(fen)
    board = chess.Board(fen)

    top_moves = STOCKFISH.get_top_moves(5)

    # Dictionary mapping SAN -> Centipawn
    moves_dict = {}
    
    # We need to generate SAN for the engine moves relative to the current board state
    for entry in top_moves:
        uci_move = entry['Move']
        move_obj = chess.Move.from_uci(uci_move)
        if move_obj in board.legal_moves:
            san_move = board.san(move_obj)
            moves_dict[san_move] = entry['Centipawn']

    # Formatted string for prompt
    formatted_moves = ' '.join([f'{m} ({cp})' for m, cp in moves_dict.items()])
    return moves_dict, f"({formatted_moves})"

def process_node(node, board, f_out):
    """Recursively walk PGN tree"""
    raw_comment = node.comment
    cleaned_comment = clean_comment(raw_comment)
    
    if cleaned_comment and len(cleaned_comment) > MIN_LEN_COMMENT:
        fen = board.fen()
        move_san = node.san()
        
        best_move_dict, best_move_str = get_top_k_best_moves(fen)
        
        # Only process if the move is actually considered good/decent by Stockfish
        # or if you strictly want to follow the text regardless of engine eval.
        # Current logic: If move not in top 5, skip.
        if move_san in best_move_dict: 
            
            # Construct messages list instead of raw string
            messages = construct_messages(
                fen,
                get_prev_k_moves(board),
                best_move_str,
                move_san,
                cleaned_comment
            )

            batch_buffer.append({
                "messages": messages,
                "fen": fen,
                "move": move_san,
                "raw_comment": cleaned_comment
            })

            if len(batch_buffer) >= BATCH_SIZE: 
                process_batch(f_out)

    if node.move in board.legal_moves: 
        board.push(node.move)
    else: 
        return # illegal move

    # Main line recursion
    if node.next():
        process_node(node.next(), board, f_out)
    
    # Variations recursion
    for variation in node.variations:
        board.pop() # Undo main move
        
        if variation.move in board.legal_moves:
            board.push(variation.move)
            if variation.next():
                process_node(variation.next(), board, f_out)
            board.pop()
        
        board.push(node.move) # Restore main move

    board.pop()

def main():
    print(f'Starting pipeline. Batch Size: {BATCH_SIZE}')
    print(f'Using Model: {MODEL_ID}')
    start_time = time.time()
    
    # update to skip n lines of data, good for restarts, just command f last commentary in outputfile
    ignore_start_lines = 200000 
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with open(PGN_FILE, 'r', encoding='utf-8') as pgn_file:
            for _ in range(ignore_start_lines): next(pgn_file)
            game_count = 0
            
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception as e:
                    print(f"Error reading game: {e}")
                    break

                if game is None: break 
                
                game_count += 1
                if game_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f'Parsed Game #{game_count} | Time elapsed: {elapsed:.2f}s')

                board = game.board()
                
                if game.next(): 
                    process_node(game.next(), board, f_out)
            
            if batch_buffer:
                process_batch(f_out)

if __name__ == "__main__":
    main()