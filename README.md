# Overview
Multimodally align a chess next move predictor with an LLM to be able to reason about potential strong next moves.

# How to run
Architecture.py is the main file to train the model. The current config trains on datasets/dataset_USETHIS.jsonl which combines data extracted from datasets/extract_traces_json.py and datasets/extract_traces_rule_based.py.

Architecture.py specifies the nn.Module subclass which combined the trained chess encoder in chess_enc_ckpt with deepseek-ai/DeepSeek-R1-Distill-Llama-8B; a Dataset class; a collator; and main which runs alignment and SFT saving in model_ckpts.

# Directory Sructure
 - datasets: code to extract reasoning data from 
    - extract_traces_rule_based.py
    - extract_traces_json.py: Use lichess studies and stockfish to prompt llm for json formatted move reasoning traces.
    - extract_traces_nl.py: Above but not forcing constrained representation -- suboptimal.
    - convert_json_to_reasoning.py: take the json formatted move format and convert to natural language for tuning. Write to jsonl.
    - prompt_iterations.py: Some early iterations on prompts used in trace extraction.
    - dataset_no_board.jsonl: reasoning_str doesnt include string board represention.
    - dataset_with_board.jsonl: reasoning_str has string board representation. -- doesnt encourage alignment.
    - reasoning_traces.txt: example trace output.
    - game_studies_new.pgn: lichess studies in PGN format - gitignored as >100MB.

 - architecture.py: align and tune llm with chess encoding prefix tokens
 - chess_encoder.py: specification for chess encoder 
 - inference.py: run inference on checkpoint in model_ckpts. Also runs experiment for fen and board representation extraction
 - model_ckpts: checkpoints after alignment, sft stored here - gitignored
 - chess_enc_ckpt: chess encoder .pt - gitignored