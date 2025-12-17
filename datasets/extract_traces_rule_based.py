import chess
from stockfish import Stockfish
from concurrent.futures import ProcessPoolExecutor
import time
import os
import tqdm
import pandas as pd
import random
import json

CENTRAL_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
KING_ADJACENT_OFFSETS = [-9, -8, -7, -1, 1, 7, 8, 9]

STOCKFISH_PATH='/users/PAS3150/jacktaylor/trace_gen/Stockfish/src/stockfish'
stockfish_engine = None

def init_worker(path):
    global stockfish_engine
    stockfish_engine = Stockfish(path=path)
    stockfish_engine.set_depth(10) #much more efficient, 10 still very deep
    stockfish_engine.update_engine_parameters({'Hash': 16, 'Threads': 1})

def detect_absolute_pin(board, move):
    board.push(move)
    attacker_sq = move.to_square
    attacker = board.piece_at(attacker_sq)
    color = attacker.color

    pinned = []

    for target_sq in board.attacks(attacker_sq):
        target = board.piece_at(target_sq)
        if not target or target.color == color: continue

        ray = chess.SquareSet.ray(attacker_sq, target_sq)
        for sq in ray:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.KING and piece.color != color:
                pinned.append(
                    f"pins {target.symbol().upper()} on {chess.square_name(target_sq)} to the king"
                )

    board.pop()
    return pinned


def is_development_move(board, move):
    piece = board.piece_at(move.from_square)
    rank = chess.square_rank(move.from_square)
    if not piece: return False
    if piece.piece_type not in (chess.KNIGHT, chess.BISHOP): return False
    if piece.color == chess.WHITE and rank != 0: return False
    if piece.color == chess.BLACK and rank != 7: return False

    return True


def get_move_info(board: chess.Board, move: chess.Move) -> str:
    if board.is_castling(move): return "castles king"

    piece = board.piece_at(move.from_square)
    piece_name = piece.symbol().upper()
    to_sq_name = chess.square_name(move.to_square)

    facts = {"check": [], "capture": [], "pin": [], "attack": [], "control": [], "develop": []}

    #capture
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured and captured.color != piece.color:
            facts["capture"].append(f"captures {captured.symbol().upper()}")

    #check
    board.push(move)
    if board.is_check(): facts["check"].append("gives check, forcing a response")

    #attacks or controls
    for sq in board.attacks(move.to_square):
        target = board.piece_at(sq)
        sq_name = chess.square_name(sq)
        if target and target.color != piece.color:
            facts["attack"].append(f"attacks {target.symbol().upper()} on {sq_name}")
        elif not target:
            if sq in CENTRAL_SQUARES:
                facts["control"].append(f"controls central square {sq_name}")
            else:
                king_sq = board.king(not piece.color)
                if king_sq is not None and abs(sq - king_sq) in KING_ADJACENT_OFFSETS:
                    facts["control"].append(f"controls square near the king {sq_name}")
    board.pop()

    facts['pin'].extend(detect_absolute_pin(board, move)) #pins
    if is_development_move(board, move): facts['develop'].append('develops a piece') #development

    #order by importance
    ordered = facts["check"] + facts["capture"] + facts["pin"] + facts["attack"] + facts["control"] + facts["develop"]
    ordered = list(dict.fromkeys(ordered))[:2]

    if not ordered: return ''

    return f"{piece_name} to {to_sq_name}: " + ', '.join(ordered)


def process_position(
    fen: str,
    depth: int = 2,
    top_k_moves: int = 3, #branch factor by top k moves
    path=None
):
    if depth < 1: return []
    if path is None: path = []

    stockfish_engine.set_fen_position(fen)
    board = chess.Board(fen)
    traces = []

    top_moves = stockfish_engine.get_top_moves(top_k_moves)
    for move_dict in top_moves:
        uci = move_dict['Move']
        move = chess.Move.from_uci(uci)
        san = board.san(move)

        eval_str = f"{move_dict['Centipawn']/100:.2f}" if move_dict.get('Centipawn') is not None else f"mate {move_dict.get('Mate')}"
        
        explanation = get_move_info(board=board, move=move)
        if not explanation: continue

        new_path = path + [f"{san} (eval {eval_str})"]
        traces.append({
            'move_path': new_path,
            'explanation': explanation
        })


        next_board = board.copy()
        next_board.push(move)

        traces.extend(
            process_position(
                fen = next_board.fen(),
                depth=depth-1,
                top_k_moves=top_k_moves,
                path=new_path
            )
        )

    return traces


def process_wrapper(fen, depth, top_k):
    traces = process_position(fen, depth, top_k)

    trace_str = ''
    for trace in traces:
        trace_str += ' -> '.join(trace['move_path'])
        trace_str += f"\n\tExplanation: {trace['explanation']}\n"

    return {
        'fen': fen,
        'trace': trace_str,
        'board': str(chess.Board(fen))
    }

def main():
    # fens = ['rnbqk2r/pp4pp/2pbpn2/3p1p2/2PP4/1P3NP1/P3PPBP/RNBQ1RK1 b kq - 0 7'] * 100
    df = pd.read_csv('lichess_puzzles.csv')
    fens = df['FEN']
    num_samples = len(fens)

    random_depths = [random.randint(2,3) for _ in range(num_samples)]
    random_top_ks = [random.randint(2,4) for _ in range(num_samples)]

    max_workers = os.cpu_count() - 1 
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(STOCKFISH_PATH,)) as executor:
        futures = executor.map(process_wrapper, fens, random_depths, random_top_ks)
        
        for trace_list in tqdm.tqdm(futures, total=num_samples): 
            results.append(trace_list)
    
    with open('train_chess_reasoning.jsonl', 'w') as f:
        for item in results:
            json_record = json.dumps(item)
            f.write(json_record + '\n')

if __name__ == '__main__':
    main()