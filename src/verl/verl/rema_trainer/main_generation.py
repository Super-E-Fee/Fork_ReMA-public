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
Generate responses given a dataset of prompts
"""
from typing import Dict, List
import ray
import numpy as np
import hydra
import os
import torch
from tqdm import tqdm
from verl.utils import hf_tokenizer
import pdb

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_rema_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from prompt.math.mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT
from prompt import FINISH_FLAG


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN"
            }
        })

    ray.get(main_task.remote(config))


def _generate_batch(
    role: str,
    history: List[List[Dict[str, str]]],
    wg: RayWorkerGroup,
    batch_ds,
    tokenizer: AutoTokenizer,
    config,
    batch_idx,
    num_batch,
    dispatch_dp_size,
) -> DataProto:
    assert role in ["meta_thinking", "reasoning"]
    system_prompt = MTA_SYSTEM_PRMOPT if role == "meta_thinking" else RA_SYSTEM_PRMOPT
    chat_lst = [[{
        "role": "system",
        "content": system_prompt
    }] for _ in range(len(batch_ds))]
    batched_questions = batch_ds["question"].tolist()
    assert len(history) == len(batch_ds)
    for i in range(len(batch_ds)):
        if role == "meta_thinking":
            chat_lst[i].append({
                "role": "user",
                "content": batched_questions[i]
            })
            for j in range(len(history[i])):
                if j % 2 == 0:
                    chat_lst[i].append({
                        "role": "assistant",
                        "content": history[i][j]["content"]
                    })
                else:
                    chat_lst[i].append({
                        "role": "user",
                        "content": history[i][j]["content"]
                    })
        else:
            chat_lst[i].append({
                "role":
                "user",
                "content":
                f'Question:\n{batched_questions[i]}\n\nInstruction:\n{history[i][0]["content"]}',
            })
            for j in range(1, len(history[i])):
                if j % 2 == 0:
                    chat_lst[i].append({
                        "role": "user",
                        "content": history[i][j]["content"]
                    })
                else:
                    chat_lst[i].append({
                        "role": "assistant",
                        "content": history[i][j]["content"]
                    })
    inputs = tokenizer.apply_chat_template(
        chat_lst,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        max_length=config.rollout.prompt_length,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = compute_position_id_with_mask(attention_mask)

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    data = DataProto.from_dict(batch_dict)

    real_batch_size = data.batch["input_ids"].shape[0]
    data = _pad_data(data, dispatch_dp_size)
    batch_size = data.batch["input_ids"].shape[0]
    assert (
        batch_size % dispatch_dp_size == 0
    ), f"batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}"

    output = wg.generate_sequences(data)
    # remove dummy data
    output = output[:real_batch_size]
    output_text = tokenizer.batch_decode(
        output.batch["input_ids"][:, -config.rollout.response_length:],
        skip_special_tokens=False,
    )

    # Remove padding and EOS tokens from the output in one pass
    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token
    output_text_clean = [
        text.replace(pad_token, "").replace(eos_token, "")
        for text in output_text
    ]

    for hist, text in zip(history, output_text_clean):
        hist.append({"role": role, "content": text})

    return output_text_clean, history, chat_lst


def _pad_data(data: DataProto, dispatch_dp_size: int):
    real_batch_size = data.batch.batch_size[0]
    if real_batch_size % dispatch_dp_size != 0:
        dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
        if dummy_data_size <= real_batch_size:
            dummy_data = data[:dummy_data_size]
        else:
            dummy_data = data.repeat(-(-dummy_data_size //
                                       real_batch_size))[:dummy_data_size]
        data = DataProto.concat([data, dummy_data])
        print(
            f"real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data"
        )
    return data


def _pad_history(input_historys: List[List[Dict[str, str]]],
                 max_length: int,
                 pad_value={
                     "role": "padding",
                     "content": "<PAD>"
                 }):
    padded_history = []
    for history in input_historys:
        current_length = len(history)
        pad_length = max(0, max_length - current_length)
        padded_history.append(history + [pad_value] * pad_length)
    return padded_history


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(
        config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."

    if not config.multi_agent.parameter_sharing:
        assert config.multi_agent.mta_model != config.multi_agent.ra_model, \
            "When parameter sharing, mta_model and ra_model must be different."

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    # dataset = dataset[:2 * config.data.batch_size]

    # ========== Initialize MTA ==========
    wg_dict = {}
    tokenizer_dict = {}
    config.model.path = config.multi_agent.mta_model
    local_path = copy_to_local(config.model.path)
    tokenizer = hf_tokenizer(local_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] *
        config.trainer.nnodes,
        name_prefix="mta_group"
        if not config.multi_agent.parameter_sharing else None,
        max_colocate_count=1)
    wg = RayWorkerGroup(resource_pool=resource_pool,
                        ray_cls_with_init=ray_cls_with_init)
    wg.init_model()
    wg_dict["meta_thinking"] = wg
    tokenizer_dict["meta_thinking"] = tokenizer

    if not config.multi_agent.parameter_sharing:
        # ========== Initialize RA ==========
        config.model.path = config.multi_agent.ra_model
        local_path = copy_to_local(config.model.path)
        tokenizer = hf_tokenizer(local_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=config,
            role="rollout")
        resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] *
            config.trainer.nnodes,
            name_prefix="ra_group",
            max_colocate_count=1)
        wg = RayWorkerGroup(resource_pool=resource_pool,
                            ray_cls_with_init=ray_cls_with_init)
        wg.init_model()
        wg_dict["reasoning"] = wg
        tokenizer_dict["reasoning"] = tokenizer
    else:
        wg_dict["reasoning"] = wg_dict["meta_thinking"]
        tokenizer_dict["reasoning"] = tokenizer_dict["meta_thinking"]

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    history_lst = [[] for _ in range(config.data.n_samples)]
    num_turn_lst = [[] for _ in range(config.data.n_samples)]
    finish_reason_lst = [[] for _ in range(config.data.n_samples)]
    meta_thinking_conversation_history_lst = [
        [] for _ in range(config.data.n_samples)
    ]
    reasoning_conversation_history_lst = [[]
                                          for _ in range(config.data.n_samples)
                                          ]

    for batch_idx in tqdm(range(num_batch)):
        print(f"[{batch_idx+1}/{num_batch}] Start to process.")
        batched_data = dataset[batch_idx * config_batch_size:(batch_idx + 1) *
                               config_batch_size]
        for i_sample in range(config.data.n_samples):
            if config.multi_agent.parameter_sharing:
                # batched_data = DataProto.from_dict(tensors={}, non_tensors=batched_data.to_dict())
                dummy_tensor = torch.arange(len(batched_data))
                batched_proto = DataProto.from_dict(
                    tensors={'dummy_tensor': dummy_tensor},
                    non_tensors={
                        'question': batched_data['question'].tolist()
                    },
                    meta_info={
                        'agent_roles': ['meta_thinking', 'reasoning'],
                        'finish_flag': FINISH_FLAG,
                        'system_prompts': {
                            'meta_thinking': MTA_SYSTEM_PRMOPT,
                            'reasoning': RA_SYSTEM_PRMOPT
                        },
                    })
                real_batch_size = dummy_tensor.shape[0]
                batched_proto = _pad_data(batched_proto, dispatch_dp_size)
                output = wg.multi_turn_generate_sequences(batched_proto)
                output = output[:real_batch_size]
                non_tensor_batch = output.non_tensor_batch
                output_lst[i_sample].extend(non_tensor_batch["response"])
                history_lst[i_sample].extend(non_tensor_batch["history"])
                num_turn_lst[i_sample].extend(non_tensor_batch["num_turns"])
                finish_reason_lst[i_sample].extend(
                    non_tensor_batch["finish_reason"])
                meta_thinking_conversation_history_lst[i_sample].extend(
                    non_tensor_batch["meta_thinking_conversation_history"])
                reasoning_conversation_history_lst[i_sample].extend(
                    non_tensor_batch["reasoning_conversation_history"])
            else:
                finish_flags = np.zeros(len(batched_data), dtype=bool)
                history = [[] for _ in range(len(batched_data))]
                finish_reason = [None for _ in range(len(batched_data))]
                meta_thinking_chat = [[] for _ in range(len(batched_data))]
                reasoning_chat = [[] for _ in range(len(batched_data))]
                for i_turn in range(config.rollout.max_num_turns):
                    # Filter out finished samples to only process unfinished ones
                    unfinished_indices = np.where(~finish_flags)[0]
                    process_data = batched_data.iloc[
                        unfinished_indices].reset_index(drop=True)
                    process_history = [
                        history[idx] for idx in unfinished_indices
                    ]

                    print(
                        f"[{batch_idx+1}/{num_batch}] turn={i_turn+1}: Processing {len(process_data)}/{len(batched_data)} samples"
                    )

                    print(
                        f"[{batch_idx+1}/{num_batch}] turn={i_turn+1} role=meta_thinking: Start to generate."
                    )
                    mta_output, process_history, meta_thinking_chat_lst = _generate_batch(
                        "meta_thinking",
                        process_history,
                        wg_dict["meta_thinking"],
                        process_data,
                        tokenizer_dict["meta_thinking"],
                        config,
                        batch_idx,
                        num_batch,
                        dispatch_dp_size,
                    )

                    if i_turn == config.rollout.max_num_turns - 1:
                        EOS_TOKEN = tokenizer_dict["meta_thinking"].eos_token
                        for i_mta in range(len(mta_output)):
                            if FINISH_FLAG not in mta_output[i_mta]:
                                # mta_txt = process_history[i_mta][-1]["content"]
                                # mta_txt = mta_txt.replace(EOS_TOKEN, "")
                                # mta_txt += " " + FINISH_FLAG + EOS_TOKEN
                                # process_history[i_mta][-1]["content"] = mta_txt
                                finish_reason[unfinished_indices[
                                    i_mta]] = "reach_max_turn"

                    # Update history for unfinished samples
                    print(
                        f"[{batch_idx+1}/{num_batch}] turn={i_turn+1} role=reasoning: Start to generate."
                    )
                    ra_output, process_history, reasoning_chat_lst = _generate_batch(
                        "reasoning",
                        process_history,
                        wg_dict["reasoning"],
                        process_data,
                        tokenizer_dict["reasoning"],
                        config,
                        batch_idx,
                        num_batch,
                        dispatch_dp_size,
                    )

                    # Update history for unfinished samples
                    for i_ui, idx in enumerate(unfinished_indices):
                        history[idx] = process_history[i_ui]
                        meta_thinking_chat[idx] = meta_thinking_chat_lst[i_ui]
                        reasoning_chat[idx] = reasoning_chat_lst[i_ui]

                    # Update finish flags if FINISH_FLAG appears in meta_thinking output
                    for i_mta, output in enumerate(mta_output):
                        if FINISH_FLAG in output:
                            finish_flags[unfinished_indices[i_mta]] = True

                    # If all samples have finish flags set, break the turn loop
                    if finish_flags.all():
                        print(
                            f"[{batch_idx+1}/{num_batch}] All samples finished at turn={i_turn+1}"
                        )
                        break
                # Only store the ra_output in the last turn, as the final answer to be checked later
                output_lst[i_sample].extend(
                    [h[-1]['content'] for h in history])
                history = _pad_history(history, 2 * config.rollout.max_num_turns)
                history_lst[i_sample].extend(history)
                num_turn_lst[i_sample].extend([len(h) / 2 for h in history])
                finish_reason_lst[i_sample].extend(finish_reason)
                meta_thinking_chat = _pad_history(meta_thinking_chat, 2 * config.rollout.max_num_turns)
                reasoning_chat = _pad_history(reasoning_chat, 2 * config.rollout.max_num_turns)
                meta_thinking_conversation_history_lst[i_sample].extend(
                    meta_thinking_chat)
                reasoning_conversation_history_lst[i_sample].extend(
                    reasoning_chat)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
    history_lst = np.array(history_lst, dtype=object)
    print(history_lst.shape)
    if len(history_lst.shape) == 3:
        history_lst = np.transpose(history_lst, axes=(1, 0, 2)).tolist()
    elif len(history_lst.shape) == 2:
        history_lst = np.transpose(history_lst, axes=(1, 0)).tolist()
    else:
        raise RuntimeError(f"history_lst.shape={history_lst.shape}")

    num_turn_lst = np.array(num_turn_lst, dtype=object)
    num_turn_lst = np.transpose(num_turn_lst, axes=(1, 0)).tolist()
    finish_reason_lst = np.array(finish_reason_lst, dtype=object)
    finish_reason_lst = np.transpose(finish_reason_lst, axes=(1, 0)).tolist()
    meta_thinking_conversation_history_lst = np.array(
        meta_thinking_conversation_history_lst, dtype=object)
    reasoning_conversation_history_lst = np.array(
        reasoning_conversation_history_lst, dtype=object)

    print(meta_thinking_conversation_history_lst.shape)
    print(reasoning_conversation_history_lst.shape)
    if len(meta_thinking_conversation_history_lst.shape) == 3:
        meta_thinking_conversation_history_lst = np.transpose(
            meta_thinking_conversation_history_lst, axes=(1, 0, 2)).tolist()
        reasoning_conversation_history_lst = np.transpose(
            reasoning_conversation_history_lst, axes=(1, 0, 2)).tolist()
    elif len(meta_thinking_conversation_history_lst.shape) == 2:
        meta_thinking_conversation_history_lst = np.transpose(
            meta_thinking_conversation_history_lst, axes=(1, 0)).tolist()
        reasoning_conversation_history_lst = np.transpose(
            reasoning_conversation_history_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset[f"responses"] = output_lst
    dataset[f"history"] = history_lst
    dataset[f"num_turns"] = num_turn_lst
    dataset[f"finish_reason"] = finish_reason_lst
    dataset[
        f"meta_thinking_conversation_history"] = meta_thinking_conversation_history_lst
    dataset[
        f"reasoning_conversation_history"] = reasoning_conversation_history_lst
    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    return 0


if __name__ == "__main__":
    main()
