# Overview
Multimodally align a chess next move predictor with an LLM to be able to reason about potential strong next moves

# Directory Sructure
 - datasets: code to extract reasoning data from 
    - extract_traces_json.py: Use lichess studies and stockfish to prompt llm for json formatted move reasoning traces.
    - extract_traces_nl.py: Above but not forcing constrained representation -- suboptimal.
    - convert_json_to_reasoning.py: take the json formatted move format and convert to natural language for tuning. Write to jsonl.
    - prompt_iterations.py: Some early iterations on prompts used in trace extraction.
    - dataset_no_board.jsonl: reasoning_str doesnt include string board represention.
    - dataset_with_board.jsonl: reasoning_str has string board representation. -- doesnt encourage alignment.
    - reasoning_traces.txt: example trace output.
    - game_studies_new.pgn: lichess studies in PGN format - gitignored as >100MB.
 - model_ckpts: checkpoints after alignment, sft stored here - gitignored
 - chess_enc_ckpt: chess encoder .pt - gitignored
 - architecture.py: align and tune llm with chess encoding prefix tokens
 - chess_encoder.py: specification for chess encoder 