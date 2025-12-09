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

MODEL_ID = 'openai/gpt-oss-20b' #'meta-llama/Llama-3.1-8B-Instruct' #'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
BATCH_SIZE = 16
MAX_NEW_TOKENS = 512 
PGN_FILE = 'game_studies_new.pgn'
OUTPUT_FILE = 'train_chess_reasoning_gptoss.jsonl'
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
    return f"""### Instruction:
You are a strong chess analyst. Given a chess position and some study commentary, explain the candidate moves 
and the reasoning just like a human annotator would — verbal, intuitive, high-level, 
but still concrete. Discuss pros/cons of several options before arriving at a final choice.

Explain your thinking the way a human chess player does when speaking out loud.
Describe ideas, rejected moves, and concerns explicitly.

Then give the best move in SAN at the end in the format MOVE: Nxd7.

Be explicit and descriptive, like a human explaining their ideas in natural language.

You will be given: 
The current position (FEN string)
The previous moves (SAN format)
The top moves as determined by the Stockfish engine (SAN format, with associated Centipawn score (positive good for white, negative good for black))
The move to justify, as determined by commentary (SAN format)
The study commentary (hidden explaination into move quality)

For your output, generate reasoning traces like you are considering the a next move from
the Stockfish suggestions, or the Move to Justify. Explain the moves strengths and weaknesses. 
Do this for all of the top moves, then pick the optimal move as determined by the commentary.

Explain briefly (max 300 words) and produce final move at the end. 
Do not waste tokens breaking down the fen string.
Pay close attention to following the few shot examples output format.

### Few-Shot Examples:
**Input**
Current Position (FEN): r1b1k2r/ppp2p1p/2n1pp2/3q4/3P4/2P2N2/P1PQ1PPP/R3KB1R b KQkq - 3 9
Previous Moves: N/A
Top Moves (Stockfish) (Rg8 (-32) Bd7 (-7) Qe4+ (5) Qa5 (21) e5 (27))
Move to Justify: Rg8
Study Commentary (Hidden Knowledge): Move selection algorithm:1 - What is the opponent's idea? He has some potential jumps like Qh6 or Qf4, but at the end of day it's very likely he will try to continue the development - and natural moves like Be2 or Bd3 comes to mind. Do we have any power to prevent it?Rg8 places the rook on an open file and the bishop can't move due to R:g2. g3 with the idea of Bg2 is not possible at the moment as well, because of the f3 knight. White is facing some coordination issues.

**Output**
I wonder about Qe4+. Very forcing and attractive — a centralizing check that probes White's king and can pick up concessions. However, it also trades into simplification lines that relieve White's cramped position (after possible Qxe4+ exchanges) and may let White escape the worst of the pressure.
Wait, I could play Bd7. Simple development that challenges White's queen and completes a piece's scope. Solid and low-risk, but slightly passive — it doesn't change the dynamic immediately or create concrete threats.
Alternatively, lets look at Qa5. Active and somewhat annoying: it eyes the queenside pawn and keeps the queen on an aggressive square. It risks overextending the queen when the center can open; it's more of a probing move than a decisive solution.
Hmm, I what about Rg8. A quiet, practical move: bring the rook into play along the g-file, keep the kingside intact and prepare to contest the g-file or double rooks. It avoids immediate simplifications and keeps pressure on White to solve problems with development and king safety.
Another option could be e5. A central break that seeks counterplay and space. Energetic, but it loosens squares and could be premature before finishing development or ensuring king safety.

From these options I like Rg8 the best.

MOVE: Rg8.

### Task Input:
Current Position (FEN): {fen}
Previous Moves: {history_str}
Top Moves (Stockfish) {best_moves}
Move to Justify: {move_san}
Study Commentary (Hidden Knowledge): "{raw_comment}"
"""


def process_batch(f):
    global batch_buffer
    if not batch_buffer: return

    prompts = [item['prompt'] for item in batch_buffer]
       
    formatted_prompts = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        formatted_prompts.append(TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    inputs = TOKENIZER(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=MAX_NEW_TOKENS*2
    ).to(MODEL.device)

    with torch.no_grad():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id
        )

    #to slice off prompt
    prompt_lengths = [len(inputs.input_ids[i]) for i in range(len(prompts))]
    
    decoded_outputs = []
    for i, output_ids in enumerate(generated_ids):
        #slice new tokens
        new_tokens = output_ids[prompt_lengths[i]:]
        text = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
        decoded_outputs.append(text)

    for i, reasoning in enumerate(decoded_outputs):
        if "REJECT" not in reasoning and len(reasoning) > 10:
            entry = {
                "fen": batch_buffer[i]['fen'],
                "move": batch_buffer[i]['move'],
                "reasoning_trace": reasoning,
                "original_comment": batch_buffer[i]['raw_comment']
            }
            f.write(json.dumps(entry) + '\n')
        else: print('REJECTED')
    
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