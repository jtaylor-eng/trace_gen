import json
import random 
from tqdm import tqdm

#input output paths
JSONL_INPUT = './subset.jsonl' 
NL_OUTPUT = './reasoning_traces.txt'

PREFIXES = [ #move proposition prefix (PREFIXES[i] + ' c5')
    'What about this move: ',
    'Hmm, what if I consider ',
    "But, let's consider playing ",
    'How about the move ',
    "Let's think about moving ",
    'Wait, how about '
]
PRO_CON_JOINS = [' and also ', ' and ', ' in addition to '] #split two pros (... good central contral and develops knight ...)
PRO_CON_NEGATIONS = [' however ', ' but '] #split pros list and cons list (... good central control however poor knight positioning ...)


def process_one_reasoning_trace(
    trace,
    think_tokens = ('<think>', '</think>')
):
    formatted_output = [think_tokens[0]]
    
    candidates = trace['candidates'] 
    random.shuffle(candidates) #shuffle as best move often first

    for cand in candidates:        
        prefix = random.choice(PREFIXES)
        
        def join_reasons(items):
            if not items: return ''
            if len(items) == 1: return items[0]
            
            text = items[0]
            for item in items[1:]:
                sep = random.choice(PRO_CON_JOINS)
                text += f'{sep}{item}'
            return text

        pro_text = join_reasons(cand['pros'])
        con_text = join_reasons(cand['cons'])
        
        sentence_parts = [f"{prefix}{cand['SAN']}. This"]
        
        if pro_text: sentence_parts.append(pro_text)
            
        if con_text:
            if pro_text:
                negation = random.choice(PRO_CON_NEGATIONS)
                sentence_parts.append(f"{negation.strip()} {con_text}")
            else:
                sentence_parts.append(con_text)
                
        full_line = ' '.join(sentence_parts) + '.'
        formatted_output.append(full_line)

    chosen_reasoning = trace['chosen_move_reasoning']
    formatted_output.append(chosen_reasoning)
        
    formatted_output.append(think_tokens[1])
    
    output = '\n'.join(formatted_output)
    # print(output) ; exit()
    return output


def process_one_json(line):
    reasoning_trace = line['reasoning_trace']
    try:
        reasoning_trace_dict = json.loads(reasoning_trace)
        reasoning_str = process_one_reasoning_trace(reasoning_trace_dict)
    except Exception as e:
        print(f'Error: {e}')
        return ''
    
    return reasoning_str + line['move']


def main():
    with open(NL_OUTPUT, 'a') as out_f:
        with open(JSONL_INPUT, 'r') as in_f:
            for line in tqdm(in_f):
                json_object = json.loads(line.strip())
                out = process_one_json(json_object)
                if out is None: continue
                
                out_f.write(out + '\n\n\n')

if __name__ == '__main__':
    main()


#for reference
EXAMPLE_IN="""
{
  "candidates": [
    {
      "SAN": "c4",
      "pros": [
        "Gains material",
        "Weakens White's pawn structure"
      ],
      "cons": [
        "Creates pawn islands"
      ]
    },
    {
      "SAN": "Be6",
      "pros": [
        "Develops bishop",
        "Supports kingside"
      ],
      "cons": [
        "Does not address material imbalance"
      ]
    },
    {
      "SAN": "Nh6",
      "pros": [
        "Develops knight",
        "Supports kingside"
      ],
      "cons": [
        "Does not address material imbalance"
      ]
    },
    {
      "SAN": "Qh4",
      "pros": [
        "Attacks White's position",
        "Develops queen"
      ],
      "cons": [
        "Does not address material imbalance"
      ]
    },
    {
      "SAN": "Qb6",
      "pros": [
        "Supports queenside",
        "Develops queen"
      ],
      "cons": [
        "Does not address material imbalance"
      ]
    }
  ],
  "chosen_move_reasoning": "The choice of c4 is a bold and aggressive move that takes advantage of the current material imbalance. By playing c4, Black gains a pawn and weakens White's pawn structure, setting up potential long-term threats. This move is a testament to the power of dynamic play and the importance of seizing opportunities to gain an advantage."
}
"""
EXAMPLE_OUT = """
<think>
How about the move c4. This Gains material and also Weakens White's pawn structure but Creates pawn islands.
Hmm, what if I consider Be6. This Develops bishop and Supports kingside however Does not address material imbalance.
Hmm, what if I consider Nh6. This Develops knight and Supports kingside but Does not address material imbalance.
How about the move Qh4. This Attacks White's position in addition to Develops queen but Does not address material imbalance.
How about the move Qb6. This Supports queenside in addition to Develops queen however Does not address material imbalance.
The choice of c4 is a bold and aggressive move that takes advantage of the current material imbalance. By playing c4, Black gains a pawn and weakens White's pawn structure, setting up potential long-term threats. This move is a testament to the power of dynamic play and the importance of seizing opportunities to gain an advantage.
</think>
"""