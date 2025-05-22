import jsonlines
from pathlib import Path

data_dir = Path(__file__).parent

raw_data_path = data_dir / 'train.jsonl'
target_data_path = data_dir / 'train_processed.jsonl'

# Following src/verl/examples/data_preprocess/math_dataset.py
# Build prompt as: question + ' ' + instruction_following
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

# Build LlamaFactory Data Format
# Use only 'instruction' and 'output'
obj_to_write = []
with jsonlines.open(raw_data_path, 'r') as reader:
    for line in reader:
        question = line['question']
        output = line['solution']

        prompt = question + ' ' + instruction_following

        data = {
            'instruction': prompt,
            'output': output
        }

        obj_to_write.append(data)

with jsonlines.open(target_data_path, 'w') as writer:
    writer.write_all(obj_to_write)

print(f'Write {len(obj_to_write)} lines to {target_data_path}')