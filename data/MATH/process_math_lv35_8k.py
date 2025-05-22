import json
import datasets 
from pathlib import Path

# https://github.com/hkust-nlp/simpleRL-reason/blob/v0/train/data/math_level3to5_data_processed_with_qwen_prompt.json
raw_data_dir = Path('./data/MATH/raw_lv35_8k')


if __name__ == '__main__':
    print(raw_data_dir)
    dataset = datasets.load_dataset(
        raw_data_dir.as_posix()
    )
    dataset = dataset['train']

    print(dataset)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('question')
            example.pop('input')
            example.pop('gt_answer')

            prompt = question + ' ' + instruction_following

            solution = example.pop('answer')
            data = {
                "data_source": 'ReMA-math',
                "prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                },
                'question': question,
                'groundtruth': solution,
                'subset': 'MATH'
            }
            return data
        return process_fn
    
    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    train_dataset.to_parquet(raw_data_dir.parent / 'train_lv3to5_8k.parquet')
    
    # load MATH500 and check not questions in MATH500 appears in train_dataset
    math_500_path = Path('./data/MATH/test500.parquet')
    math_500_dataset = datasets.load_dataset('parquet', data_files=str(math_500_path))
    math_500_dataset = math_500_dataset['train']
    
    train_questions = set(train_dataset['question'])
    math_500_questions = set(math_500_dataset['question'])
    
    not_in_train = math_500_questions - train_questions
    print(f"Number of questions in MATH500 not in train: {len(not_in_train)}")
    assert len(not_in_train) == len(math_500_dataset), f"Number of questions in MATH500 not in train: {len(not_in_train)}"