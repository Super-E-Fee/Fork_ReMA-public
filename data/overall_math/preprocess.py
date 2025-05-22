# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', required=True)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = Path(__file__).parent
    data_source = data_source / 'all_test_data.jsonl'
    print(f"Loading from {data_source}  ...", flush=True)
    
    dataset = datasets.load_dataset('json', data_files=data_source.as_posix())
    # since we only have one file, we can directly use the 'train' split
    test_dataset = dataset['train']

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('question')
            solution = example.pop('groundtruth')
            answer = example.pop('answer')
            subset = example.pop('dataset')
            data = {
                "data_source": 'ReMA-math',
                "prompt": [
                    {
                        "role": "user",
                        "content": question + instruction
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                },
                'question': question,
                'groundtruth': solution,
                'subset': subset
            }
            return data

        return process_fn

    # train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # append opencompass/AIME2025
    aime2025_dataset_part_1 = datasets.load_dataset("opencompass/AIME2025", "AIME2025-I")['test']
    aime2025_dataset_part_2 = datasets.load_dataset("opencompass/AIME2025", "AIME2025-II")['test']
    # DatasetDict({
    #     test: Dataset({
    #         features: ['question', 'answer'],
    #         num_rows: 15
    #     })
    # })
    def process_aime(example, idx):
        question = example.pop('question')
        answer = example.pop('answer')
        subset = 'aime25'
        data = {
            "data_source": 'ReMA-math',
            "prompt": [
                {
                    "role": "user",
                    "content": question + instruction
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': 'test',
                'index': idx
            },
            'question': question,
            'groundtruth': answer,
            'subset': subset
        }
        return data
    
    aime2025_dataset_part_1 = aime2025_dataset_part_1.map(function=process_aime, with_indices=True)
    aime2025_dataset_part_2 = aime2025_dataset_part_2.map(function=process_aime, with_indices=True)
    
    aime2025_dataset = datasets.concatenate_datasets([aime2025_dataset_part_1, aime2025_dataset_part_2])
    
    test_dataset = datasets.concatenate_datasets([test_dataset, aime2025_dataset])

    # train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
