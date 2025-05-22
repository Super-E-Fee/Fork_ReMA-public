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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from functools import partial
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException
from tqdm import tqdm
import hydra
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import math, gsm8k, math_verify
import pandas as pd
import numpy as np


def select_reward_fn(data_source):
    # if data_source == 'lighteval/MATH':
    #     return math.compute_score
    # elif data_source == "HuggingFaceH4/aime_2024":
    #     # return math.compute_score
    #     return math_verify.compute_score
    # elif data_source in ["HuggingFaceH4/MATH-500", 'DigitalLearningGmbH/MATH-lighteval']:
    #     # return math.compute_score
    #     return math_verify.compute_score
    # elif data_source == "openai/gsm8k":
    #     return gsm8k.compute_score
    if data_source == 'ReMA-math':
        return math_verify.compute_score
    else:
        # return math_verify.compute_score
        raise NotImplementedError(f"Unknow data_source: {data_source}")

def multi_sample_reward_fn(params):
    scores = []
    data_source, r_list, gt = params
    reward_fn = select_reward_fn(data_source)
    for r in r_list:
        score = reward_fn(r, gt)
        scores.append(score)
    return scores

@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    # prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    
    # Add dictionary to track passes and totals for each subset
    subset_stats = {}  # Format: {subset_name: {'passes': 0, 'total': 0}}

    params = [
        (data_sources[i], responses[i], reward_model_data[i]['ground_truth'])
        for i in range(total)
    ]
    scores = []
    with ProcessPool(max_workers=1) as pool:
        future = pool.map(multi_sample_reward_fn, params, timeout=10)
        iterator = future.result()
        with tqdm(total=total, desc="Computing scores") as pbar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except TimeoutError:
                    print("TimeoutError")
                    scores.append([0.0])
                except TimeoutException:
                    print("Math verify timeout execption")
                    scores.append([0.0])
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    raise e
                pbar.update(1)
    assert len(scores) == total
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        # select reward score based on data_source
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        
        # Get the subset information
        subset = dataset['subset'][i]
        
        # Initialize subset stats if not exists
        if subset not in subset_stats:
            subset_stats[subset] = {'passes': 0, 'total': 0}
        
        # Increment total count for this subset
        subset_stats[subset]['total'] += 1
        
        # score_lst = []
        # for r in response_lst:
        #     score = reward_fn(r, ground_truth)
        #     score_lst.append(score)
        score_lst = scores[i]

        max_score = np.max(score_lst)

        if max_score == 1:
            passes += 1
            # Increment pass count for this subset
            subset_stats[subset]['passes'] += 1

    # Print overall accuracy
    print(f'Overall accuracy: {passes / total:.4f} ({passes}/{total})')
    
    # Create a DataFrame for results instead of simple printing
    results_data = []
    # Add overall accuracy
    results_data.append({
        'Subset': 'Overall',
        'Accuracy': f'{passes / total:.4f}',
        'Passes': passes,
        'Total': total
    })
    
    # Add subset accuracies
    for subset, stats in subset_stats.items():
        subset_accuracy = stats['passes'] / stats['total'] if stats['total'] > 0 else 0
        results_data.append({
            'Subset': subset,
            'Accuracy': f'{subset_accuracy:.4f}',
            'Passes': stats['passes'],
            'Total': stats['total']
        })
    
    # Create DataFrame and display in a format suitable for copying to spreadsheets
    results_df = pd.DataFrame(results_data)
    print(results_df.to_csv(sep=',', index=False))

if __name__ == '__main__':
    main()
