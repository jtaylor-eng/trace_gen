import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import chess
import chess.pgn
import re
import json
import time

#Constants / config
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BATCH_SIZE = 64
MAX_NEW_TOKENS = 1024
PGN_FILE = 'game_studies_new.pgn'
OUTPUT_FILE = 'train_chess_reasoning.jsonl'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
TOKENIZER.pad_token = TOKENIZER.eos_token
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='cuda',
)
MODEL.config.use_cache = True

batch_buffer = []


def clean_comment(comment):
    comment = re.sub(r'\[%[^\]]+\]', '', comment) #remove [%...]
    comment = re.sub(r'\s+', ' ', comment).strip() #remove excess whitespace
    return comment


def construct_prompt(fen, move_san, raw_comment):
    instruction = f"""### Instruction:
You are a Chess Grandmaster playing a game. You are currently looking at the board state defined by the FEN below.

Your goal is to generate a "Thought Process" that leads to the decision to play the move **{move_san}**.
Use the provided **Study Commentary** as the ground truth for your strategic reasoning, but do not explicitly mention "the commentary" or "the study". Write the thought process in the first person ("I"), analyzing the position, candidates, and concluding why {move_san} is the best move.

If the commentary is useless (e.g. "Chapter 1", "White wins"), strictly output only the word: REJECT.

### Input Data:
**Current Position (FEN):** {fen}
**Move to Justify:** {move_san}
**Study Commentary (Hidden Knowledge):** "{raw_comment}"

### Thought Process:
"""
    return instruction


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
        max_length=2048
    ).to(MODEL.device)

    with torch.no_grad():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.5,
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
    
    f.flush()
    batch_buffer = []


def process_node(node, board, f_out):
    """Recursively walk PGN tree"""
    raw_comment = node.comment
    cleaned_comment = clean_comment(raw_comment)

    if cleaned_comment and len(cleaned_comment) > 10:
        fen = board.fen()
        move_san = node.san()
        
        #add in buffer
        batch_buffer.append({
            "prompt": construct_prompt(fen, move_san, cleaned_comment),
            "fen": fen,
            "move": move_san,
            "raw_comment": cleaned_comment
        })

        #batch full, begin processing
        if len(batch_buffer) >= BATCH_SIZE: process_batch(f_out)


    if node.move in board.legal_moves:
        board.push(node.move)
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
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception as e:
                    print(f'PGN Parsing Error: {e}')
                    continue
                
                if game is None:
                    break
                
                game_count += 1
                if game_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f'Parsed Game #{game_count} | Time elapsed: {elapsed:.2f}s')

                board = game.board()
                
                if game.next():
                    process_node(game.next(), board, f_out)

if __name__ == "__main__":
    main()