import chess
from stockfish import Stockfish

stockfish = Stockfish(path="/users/PAS3150/jacktaylor/trace_gen/Stockfish/src/stockfish")

fen_str = 'rnbqk2r/pp4pp/2pbpn2/3p1p2/2PP4/1P3NP1/P3PPBP/RNBQ1RK1 b kq - 0 7'

stockfish.set_fen_position(fen_str)
board = chess.Board(fen_str)

best_move_uci = stockfish.get_best_move()

best_move = chess.Move.from_uci(best_move_uci)

print(f"The best move is: {best_move}")

top_moves = stockfish.get_top_moves(5)
print(top_moves)

def get_top_k_best_moves(fen: str, k: int = 5):
    stockfish.set_fen_position(fen)
    board = chess.Board(fen)

    top_moves = stockfish.get_top_moves(5)

    #top i move: centipawn score
    moves_dict = {}
    for entry in top_moves:
        moves_dict[board.san(chess.Move.from_uci(entry['Move']))] = entry['Centipawn']

    return moves_dict, f"({' '.join([f'{k} ({v})' for k,v in moves_dict.items()])})"

print(get_top_k_best_moves(fen_str)[1])


