import numpy as np
from omegaconf import DictConfig
from verl import DataProto
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import collate_fn as data_proto_collate_fn, pad_dataproto_to_divisor, unpad_dataproto
import torch
import unicodedata

def normalize_text(text):
    return unicodedata.normalize('NFKC', text)

def _pad_history(input_historys: List[List[Dict[str, str]]],
                 max_length: int,
                 pad_value={
                     "role": "padding",
                     "content": "<PAD>"
                 }):
    padded_history = []
    for history in input_historys:
        current_length = len(history)
        pad_length = max_length - current_length
        assert pad_length >= 0, f"current_length: {current_length}, max_length: {max_length}"
        padded_history.append(history + [pad_value] * pad_length)
    return padded_history


def _encode_conversation(
    conversation: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    num_gen_tokens: List[int],
    stop_reasons: List[Optional[str]],
):
    IGNORE_INDEX = -100
    labels = []
    step_ids = []
    cur_len = 0
    cur_hist = []
    i_step = 0
    for i, msg in enumerate(conversation):
        if msg["role"] in ["system", "user"]:
            pass
        elif msg["role"] == "assistant":
            # query string
            query = tokenizer.apply_chat_template(cur_hist,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
            # response string
            response = msg["content"]
            query_ids = tokenizer.encode(query, add_special_tokens=True)
            query_response_ids = tokenizer.encode(query + response,
                                                  add_special_tokens=True)
            response_ids = query_response_ids[len(query_ids):]
            input_ids = query_response_ids

            ################################################################
            # input_ids:
            # | this | is | a | test | <im_end> | <im_start> | <assistant> | this | is | a | response | <im_end> |
            # query_ids:
            # | this | is | a | test | <im_end> | <im_start> | <assistant> |
            # response_ids:
            # | this | is | a | response | <im_end> |
            # step_ids:
            # |IGNORE| IG |IG | IG   | IG       | IG         | i_step      |i_step| ... |i_step| IGNORE |
            # labels:
            # |IGNORE| IG |IG | IG   | IG       | IG         | this | is   | a | response   | <im_end> | IGNORE
            #################################################################
            step_ids.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))
            labels.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))

            stop_reason = stop_reasons[i_step]
            # if stop normally, add eos token
            if stop_reason == "stop":
                labels.extend(response_ids + [tokenizer.eos_token_id])
                step_ids.extend([i_step] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = len(response_ids) + 1
            # if truncated, do not add eos token as label
            elif stop_reason == "length":
                # print("# STOP REASON:", stop_reasons[i_step])
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([i_step] * len(response_ids) + [IGNORE_INDEX])
                num_gen_tokens[i_step] = len(response_ids)
            elif stop_reason in [
                    "stop_when_truncated", "completion_token_exceeded"
            ]:
                # special case for dummy response
                # XXX: in this case, response == ""
                assert response == ""
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([IGNORE_INDEX] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = 0
                break

            i_step += 1
            cur_len = len(query_response_ids)
        else:
            raise ValueError(f"Unknown message role: {msg['role']}")
        cur_hist.append(msg)

    assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"
    return input_ids, labels, step_ids


class MultiAgentRollout:

    def __init__(
        self, 
        config: DictConfig,
        tokenizers: Dict[str, PreTrainedTokenizer],
        rollout_wg_dict: Dict[str, RayWorkerGroup]
    ):
        self.config = config
        self.tokenizers = tokenizers
        self.rollout_wg_dict = rollout_wg_dict

    def _apply_chat_template(self, chat_lst: List[List[Dict[str, str]]],
                             tokenizer: PreTrainedTokenizer):
        """Apply chat template and encode"""
        return tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def _initialize_conversation_state(self, batch_size):
        """Initialize conversation state variables"""
        history = [[] for _ in range(batch_size)]
        finish_flags = np.zeros(batch_size, dtype=bool)
        finish_reason = [None for _ in range(batch_size)]
        return history, finish_flags, finish_reason

    def _build_chat_list_for_role(
        self,
        role: str,
        history_list: List[List[Dict[str, str]]],
        questions: List[str],
        system_prompts: Dict[str, str],
        agent_roles: List[str],
    ):
        """Build chat list for a specific role"""

        chat_lst = [[{
            "role": "system",
            "content": system_prompts[role]
        }] for _ in range(len(history_list))]

        for i, (hist, question) in enumerate(zip(history_list, questions)):
            if role == agent_roles[0]: # meta-thinking
                chat_lst[i].append({"role": "user", "content": question})
                for j in range(len(hist)):
                    if j % 2 == 0:
                        chat_lst[i].append({
                            "role": "assistant",
                            "content": hist[j]["content"]
                        })
                    else:
                        chat_lst[i].append({
                            "role": "user",
                            "content": hist[j]["content"]
                        })
            else: # reasoning
                chat_lst[i].append({
                    "role":
                    "user",
                    "content":
                    f'Question:\n{question}\n\nInstruction:\n{hist[0]["content"]}',
                })
                for j in range(1, len(hist)):
                    if (j + 1) % 2 == 0:
                        chat_lst[i].append({
                            "role": "assistant",
                            "content": hist[j]["content"]
                        })
                    else:
                        chat_lst[i].append({
                            "role": "user",
                            "content": hist[j]["content"]
                        })

        return chat_lst

    def _prepare_role_prompts(
        self,
        role: str,
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        questions: List[str],
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
    ) -> Tuple[DataProto, List[List[Dict[str, str]]]]:
        """Prepare prompts for a specific role"""

        # Prepare history and questions for currently unfinished samples
        current_history = [history[idx] for idx in unfinished_indices]
        current_questions = [questions[idx] for idx in unfinished_indices]

        # Build chat list
        chat_lst = self._build_chat_list_for_role(
            role,
            current_history,
            current_questions,
            system_prompts,
            agent_roles,
        )

        # Apply chat template and encode
        inputs = self._apply_chat_template(chat_lst, tokenizers[role])
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        return data, chat_lst

    def _filter_truncated_prompts_before_generation(
        self,
        prompt_proto: DataProto,
        chat_lst: List[List[Dict[str, str]]],
        role: str,
        agent_roles: List[str],
        history: List[List[Dict[str, str]]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizer: PreTrainedTokenizer,
        unfinished_indices: np.ndarray,
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        i_turn: int,
    ):
        # check current state length
        non_trunc_input = tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=False,
            max_length=None,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        # state length
        seq_lens = non_trunc_input["attention_mask"].sum(dim=1).tolist()
        # if state length is larger than prompt length, the trajectory is terminated
        if not all([l <= self.config.prompt_length for l in seq_lens]):
            # drop the terminated trajectories
            new_seq_lens = []
            new_unfinished_indices = []
            new_prompt_protos = []
            new_chat_lst = []
            for i, idx in enumerate(unfinished_indices):
                if seq_lens[i] <= self.config.prompt_length:
                    new_unfinished_indices.append(idx)
                    new_prompt_protos.append(prompt_proto[i])
                    new_seq_lens.append(seq_lens[i])
                    new_chat_lst.append(chat_lst[i])
                else:
                    # set finish flag and finish reason
                    finish_flags[idx] = True
                    finish_reason[idx] = "completion_token_exceeded"
                    print(f"idx={idx}, completion_token_exceeded")
                    # if the next gen is for reasoning agent, we need to add a dummy response in history
                    if role == agent_roles[1]:
                        history[idx].append({
                            "role":
                            agent_roles[1],
                            "content":
                            "",
                            "num_gen_tokens":
                            0,
                            "stop_reason":
                            "completion_token_exceeded",
                        })
                        # update conversation history for reasoning agent
                        conversation_history[agent_roles[1]][idx] = chat_lst[i]
                    else:
                        if i_turn == 0:
                            raise RuntimeError(
                                f"1st round prompt larger than prompt length: {seq_lens[i]} > {self.config.prompt_length}"
                            )

            if len(new_prompt_protos):
                # collate prompt needed to generate this round
                new_prompt_proto = data_proto_collate_fn(new_prompt_protos)
                new_prompt_proto.meta_info = prompt_proto.meta_info
            else:
                new_prompt_proto = None
            return new_prompt_proto, new_chat_lst, new_unfinished_indices
        else:
            return prompt_proto, chat_lst, unfinished_indices

    def _generate_role_responses(
        self,
        rollout: RayWorkerGroup,
        prompt_proto: DataProto,
        tokenizer: PreTrainedTokenizer,
        response_length: int,
    ):
        """Generate responses for the current role"""
        pad_prompt_proto, pad_size = pad_dataproto_to_divisor(prompt_proto, rollout.world_size)
        output = rollout.raw_generate_sequences(pad_prompt_proto)
        unpad_output = unpad_dataproto(output, pad_size=pad_size)
        resp_lens = (unpad_output.batch["attention_mask"][:, -response_length:].sum(
            dim=1).tolist())
        vllm_output_text = unpad_output.non_tensor_batch["text"].tolist()

        # output_text = tokenizer.batch_decode(
        #     output.batch["input_ids"][:, -response_length:],
        #     skip_special_tokens=False,
        # )

        # # Remove padding and EOS tokens from the output in one pass
        # pad_token = tokenizer.pad_token
        # eos_token = tokenizer.eos_token
        # output_text_clean = [
        #     text.replace(pad_token, "").replace(eos_token, "")
        #     for text in output_text
        # ]

        # for i, (decode_txt, vllm_txt) in enumerate(zip(output_text_clean, vllm_output_text)):
        #     if decode_txt != vllm_txt:
        #         print(f"i={i}, decode_txt={decode_txt}, vllm_txt={vllm_txt}")

        num_gen_tokens = unpad_output.non_tensor_batch[
            "gen_response_lengths"].tolist()
        stop_reasons = unpad_output.non_tensor_batch["stop_reasons"].tolist()

        # return output_text_clean, num_gen_tokens, stop_reasons, resp_lens
        return vllm_output_text, num_gen_tokens, stop_reasons, resp_lens

    def _update_history_and_check_finish(
        self,
        role: str,
        current_outputs: List[str],
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        finish_flag: str,
        agent_roles: List[str],
        num_gen_tokens: List[int],
        stop_reasons: List[Optional[str]],
        questions: List[str],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        system_prompts: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
    ):
        """Update conversation history and check completion flags"""
        # Update history
        assert len(current_outputs) == len(
            unfinished_indices
        ), f"{len(current_outputs)} != {len(unfinished_indices)}"
        for i, idx in enumerate(unfinished_indices):
            history[idx].append({
                "role": role,
                "content": current_outputs[i],
                "num_gen_tokens": num_gen_tokens[i],
                "stop_reason": stop_reasons[i],
            })

        # Update finish flags
        # Check completion flags
        if role == agent_roles[1]:
            for i, idx in enumerate(unfinished_indices):
                last_output = history[idx][-2]
                assert last_output["role"] == agent_roles[0]
                response = last_output["content"]
                if finish_flag and finish_flag in response:
                    finish_flags[idx] = True
                    finish_reason[idx] = None

        if self.config.stop_when_truncated:
            for i, stop_reason in enumerate(stop_reasons):
                # if stop_reason == "length" and not finish_flags[unfinished_indices[i]]:
                # XXX: even if stop by finish_flag, if current output is truncated, we need
                #  mark this trajectory as terminated
                if stop_reason == "length":
                    idx = unfinished_indices[i]
                    print(f"idx={idx}, stop_when_truncated")
                    finish_flags[idx] = True
                    finish_reason[idx] = "stop_when_truncated"
                    if role == agent_roles[0]:
                        # update conversation for reasoning agent
                        _, new_conversation = self._prepare_role_prompts(
                            agent_roles[1],
                            [idx],
                            history,
                            questions,
                            agent_roles,
                            system_prompts,
                            tokenizers,
                        )
                        conversation_history[
                            agent_roles[1]][idx] = new_conversation[0]

                        # add dummy history of reasoning agent
                        history[idx].append({
                            "role":
                            agent_roles[1],
                            "content":
                            "",
                            "num_gen_tokens":
                            0,
                            "stop_reason":
                            "stop_when_truncated",
                        })

    def _run_multi_turn_conversation(
        self,
        prompts: DataProto,
        tokenizers: Dict[str, PreTrainedTokenizer],
        max_num_turns: int,
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        finish_flag: str,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        response_length: int,
    ):
        questions = prompts.non_tensor_batch["question"]
        assert len(finish_flags) == len(
            questions), f"{finish_flags.shape} != {len(questions)}"

        conversation_history = {
            role: [None for _ in range(len(questions))]
            for role in agent_roles
        }

        for i_turn in range(max_num_turns):
            # Get indices of unfinished samples
            unfinished_indices = np.where(~finish_flags)[0]
            print(f"turn {i_turn+1} of {max_num_turns}, \
                    {len(unfinished_indices)}/{len(questions)} unfinished")

            if len(unfinished_indices) == 0:
                break
            # Each role takes turns generating in every round
            for i_role, role in enumerate(agent_roles):
                print(f"role: {role}")
                # Prepare prompts for current role
                prompt_proto, chat_lst = self._prepare_role_prompts(
                    role,
                    unfinished_indices,
                    history,
                    questions,
                    agent_roles,
                    system_prompts,
                    tokenizers,
                )

                # side effect on convsersation_history and history
                prompt_proto, chat_lst, unfinished_indices = (
                    self._filter_truncated_prompts_before_generation(
                        prompt_proto=prompt_proto,
                        chat_lst=chat_lst,
                        role=role,
                        agent_roles=agent_roles,
                        history=history,
                        conversation_history=conversation_history,
                        tokenizer=tokenizers[role],
                        unfinished_indices=unfinished_indices,
                        finish_flags=finish_flags,
                        finish_reason=finish_reason,
                        i_turn=i_turn,
                    ))
                if len(unfinished_indices) == 0:
                    break

                prompt_proto.meta_info.update(prompts.meta_info)
                for i, chat in enumerate(chat_lst):
                    idx = unfinished_indices[i]
                    conversation_history[role][idx] = chat

                # Generate responses for current role
                current_outputs, num_gen_tokens, stop_reasons, resp_lens = (
                    self._generate_role_responses(
                        rollout=self.rollout_wg_dict[role],
                        prompt_proto=prompt_proto,
                        tokenizer=tokenizers[role],
                        response_length=response_length,
                    ))

                # XXX(ziyu): remove finish flag in output for reasoning agent here
                #  consider move to a post-processing function
                if role == agent_roles[1] and finish_flag:
                    current_outputs = [
                        output.replace(finish_flag, "").rstrip()
                        for output in current_outputs
                    ]

                # XXX(ziyu): side effect on `history`
                self._update_history_and_check_finish(
                    role,
                    current_outputs,
                    unfinished_indices,
                    history,
                    finish_flags,
                    finish_reason,
                    finish_flag,
                    agent_roles,
                    num_gen_tokens,
                    stop_reasons,
                    questions,
                    conversation_history,
                    system_prompts,
                    tokenizers,
                )
                unfinished_indices = np.where(~finish_flags)[0]
                if len(unfinished_indices) == 0:
                    break
        # use the last output of each agent as latest output response
        latest_outputs = [h[-1]["content"] for h in history]
        return latest_outputs, conversation_history

    def _mark_unfinished_as_max_turns(self, finish_flags: np.ndarray,
                                      finish_reason: List[Optional[str]]):
        """Mark unfinished samples as reaching maximum turns"""
        for i in range(len(finish_flags)):
            if not finish_flags[i]:
                finish_reason[i] = "reach_max_turn"

    def _build_tensor_dict(
        self,
        last_round_responses: List[Dict[str, str]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        num_gen_token_lst: Dict[str, List[List[int]]],
        stop_reason_lst: Dict[str, List[List[Optional[str]]]],
        max_num_turns: int,
        finish_reason: List[Optional[str]],
    ):
        # add last round output to make full conversation
        for i_batch in range(len(last_round_responses)):
            for role in last_round_responses[i_batch]:
                conversation_history[role][i_batch].append({
                    "role":
                    "assistant",
                    "content":
                    last_round_responses[i_batch][role],
                })

        input_ids_lst = {role: [] for role in conversation_history.keys()}
        labels_lst = {role: [] for role in conversation_history.keys()}
        step_ids_lst = {role: [] for role in conversation_history.keys()}

        # build tensors for training
        for i_batch in range(len(last_round_responses)):
            for role in conversation_history.keys():
                # encode conversation into input_ids, labels, step_ids
                # XXX(ziyu): need to consider stop reason here ?
                input_ids, labels, step_ids = _encode_conversation(
                    conversation_history[role][i_batch],
                    tokenizers[role],
                    num_gen_token_lst[role][i_batch],
                    stop_reason_lst[role][i_batch],
                )
                input_ids_lst[role].append(input_ids)
                labels_lst[role].append(labels)
                step_ids_lst[role].append(step_ids)

        # Apply padding to create tensors
        batch_size = len(last_round_responses)
        tensor_dict = {}
        finish_reason_array = []
        for fr in finish_reason:
            if fr == "reach_max_turn":
                finish_reason_array.append(1)
            elif fr == "completion_token_exceeded":
                finish_reason_array.append(2)
            elif fr == "stop_when_truncated":
                finish_reason_array.append(3)
            elif fr is None:
                finish_reason_array.append(0)
            else:
                raise ValueError(f"Unknown finish reason: {fr}")

        for role in conversation_history.keys():
            # Find max length for padding
            max_length = max([len(ids) for ids in input_ids_lst[role]])
            if max_length > self.config.response_length + self.config.prompt_length:
                print(
                    f"role: {role}, max_length={max_length} > {self.config.response_length + self.config.prompt_length}"
                )
                # raise RuntimeError(f"max_length={max_length} > {self.config.response_length + self.config.prompt_length}")

            # Use max length for padding and gathering
            max_length = self.config.response_length + self.config.prompt_length

            # Pad and convert to tensors
            padded_input_ids = torch.full((batch_size, max_length),
                                          tokenizers[role].pad_token_id,
                                          dtype=torch.long)
            padded_labels = torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long  # IGNORE_INDEX
            )
            padded_step_ids = torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long  # IGNORE_INDEX
            )
            attention_mask = torch.zeros((batch_size, max_length),
                                         dtype=torch.long)

            # Fill in the actual values
            for i, (input_ids, labels, step_ids) in enumerate(
                    zip(input_ids_lst[role], labels_lst[role],
                        step_ids_lst[role])):
                seq_len = min(len(input_ids), max_length)
                padded_input_ids[i, :seq_len] = torch.tensor(
                    input_ids[:seq_len], dtype=torch.long)
                padded_labels[i, :seq_len] = torch.tensor(labels[:seq_len],
                                                          dtype=torch.long)
                padded_step_ids[i, :seq_len] = torch.tensor(step_ids[:seq_len],
                                                            dtype=torch.long)
                attention_mask[i, :seq_len] = 1

            # Compute position ids from attention mask
            position_ids = compute_position_id_with_mask(attention_mask)

            padded_num_gen_tokens = torch.full((batch_size, max_num_turns),
                                               0,
                                               dtype=torch.long)
            for i, num_gen_tokens in enumerate(num_gen_token_lst[role]):
                padded_num_gen_tokens[i, :len(num_gen_tokens)] = torch.tensor(
                    num_gen_tokens, dtype=torch.long)
            padded_stop_reasons = torch.full((batch_size, max_num_turns),
                                             0,
                                             dtype=torch.bool)

            for i, stop_reasons in enumerate(stop_reason_lst[role]):
                stop_reason_array = np.array(
                    [0 if r == "stop" else 1 for r in stop_reasons])
                padded_stop_reasons[i, :len(stop_reason_array)] = torch.tensor(
                    stop_reason_array, dtype=torch.bool)

            # Create a separate tensor dict for each role
            tensor_dict[role] = dict(
                {
                    "input_ids": padded_input_ids,
                    "labels": padded_labels,
                    "step_ids": padded_step_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_gen_tokens": padded_num_gen_tokens,
                    "stop_reasons": padded_stop_reasons,
                    "turn_finished": torch.tensor(finish_reason_array),
                }, )

        # remove side effect
        for i_batch in range(len(last_round_responses)):
            for role in last_round_responses[i_batch]:
                conversation_history[role][i_batch].pop()

        return tensor_dict

    def _prepare_final_output(
        self,
        tensor_dict: Dict[str, Dict[str, torch.Tensor]],
        latest_outputs: List[str],
        history: List[List[Dict[str, str]]],
        finish_reason: List[Optional[str]],
        agent_roles: List[str],
        prompts: DataProto,
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
    ):
        """Prepare final output"""

        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch["finish_reason"] = finish_reason
        non_tensor_batch["num_turns"] = [
            len(h) // len(agent_roles) for h in history
        ]
        non_tensor_batch["response"] = latest_outputs

        padded_history = _pad_history(history, 2 * self.config.max_num_turns)
        padded_conversation_history = {
            role:
            _pad_history(conversation_history[role],
                         2 * self.config.max_num_turns)
            for role in agent_roles
        }

        non_tensor_batch["history"] = padded_history
        for role in agent_roles:
            non_tensor_batch[
                f"{role}_conversation_history"] = padded_conversation_history[
                    role]

        flat_tensor_dict = {}
        for role in tensor_dict.keys():
            for key in tensor_dict[role].keys():
                flat_tensor_dict[f"{role}_{key}"] = tensor_dict[role][key]

        return DataProto.from_dict(
            tensors=flat_tensor_dict,
            non_tensors=non_tensor_batch,
            meta_info=prompts.meta_info,
        )
    
    def _checking(
        self,
        history: List[List[Dict[str, str]]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        agent_roles: List[str],
        last_round_responses: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        tensor_dict: Dict[str, torch.tensor],
        final_output: DataProto,
    ):
        ###################### TESTING ######################
        # 1. test lengths of history and conversation_history
        #  len(history[i]) == len(conversation_history[role][i]) * len(agent_roles)
        for i in range(len(history)):
            assert len(history[i]) == len(conversation_history[agent_roles[0]][i]), \
                f"len(history[i]) = {len(history[i])} != len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])}"
            assert len(conversation_history[agent_roles[0]][i]) == len(conversation_history[agent_roles[1]][i]), \
                f"len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])} != len(conversation_history[agent_roles[1]][i]) = {len(conversation_history[agent_roles[1]][i])}"

        # 2. check history role name order
        for i in range(len(history)):
            for j in range(len(history[i])):
                assert history[i][j]['role'] == agent_roles[j % len(agent_roles)], \
                    f"history[i][j]['role'] = {history[i][j]['role']} != agent_roles[j % len(agent_roles)] = {agent_roles[j % len(agent_roles)]}"
                
            # 2.1 check last round response
            for i_role, role in enumerate(agent_roles):
                assert history[i][-len(agent_roles) + i_role]['role'] == role, \
                    f"history[i][-len(agent_roles) + i_role]['role'] = {history[i][-len(agent_roles) + i_role]['role']} != role = {role}"
                assert history[i][-len(agent_roles) + i_role]['content'] == last_round_responses[i][role], \
                    f"history[i][-1]['content'] = {history[i][-1]['content']} != last_round_responses[i][role] = {last_round_responses[i][role]}"
            
        # 3. check conversation_history role name order
        for i_role, role in enumerate(conversation_history.keys()):
            for i in range(len(conversation_history[role])):
                for j in range(len(conversation_history[role][i])):
                    if j == 0:
                        assert conversation_history[role][i][j]['role'] == "system", \
                            f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'system'"
                    elif j % 2 == 1:
                        assert conversation_history[role][i][j]['role'] == "user", \
                            f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'user'"
                    else:
                        assert conversation_history[role][i][j]['role'] == "assistant", \
                            f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'assistant'"
                        # check history string equals to conversation_string
                        assert conversation_history[role][i][j]['content'] == history[i][i_role + j - 2]['content'], \
                            f"'{[conversation_history[role][i][j]['content']]}' != '{[history[i][i_role + j - 2]['content']]}'"

        # 4. check input_ids
        for i_role, role in enumerate(agent_roles):
            role_tensor_dict = tensor_dict[role]
            for i in range(len(role_tensor_dict["input_ids"])):
                input_ids = role_tensor_dict["input_ids"][i]
                labels = role_tensor_dict["labels"][i]
                attention_mask = role_tensor_dict["attention_mask"][i]
                step_ids = role_tensor_dict["step_ids"][i]
                stop_reasons = role_tensor_dict["stop_reasons"][i]
                num_turn = final_output.non_tensor_batch["num_turns"][i]

                query_response = tokenizer.decode(input_ids[attention_mask == 1].tolist())
                raw_query_response = tokenizer.apply_chat_template(
                    conversation_history[role][i], 
                    add_generation_prompt=True, 
                    padding=True, 
                    truncation=False, 
                    max_length=None, 
                    tokenize=False, 
                ) + last_round_responses[i][role]
                
                assert step_ids.max() == num_turn - 1 or stop_reasons[num_turn - 1] != 0, \
                    f"{step_ids.max()} != {num_turn - 1} or {stop_reasons[num_turn - 1]} != 0"

                # FIXME: tokenizer has some issues on decode and encode unicode chars.

                assert normalize_text(query_response) == normalize_text(raw_query_response), \
                    f"'{query_response}' != '{raw_query_response}'"
                for i_turn in range(num_turn):
                    turn_labels = labels[step_ids == i_turn]
                    if stop_reasons[i_turn] == 0:
                        assert turn_labels[-1] == tokenizer.eos_token_id
                        turn_labels = turn_labels[:-1] # drop eos
                    response = tokenizer.decode(turn_labels.tolist())
                    assert normalize_text(response) == normalize_text(history[i][i_role + i_turn * len(agent_roles)]['content']), \
                        f"'{response}' != '{history[i][i_role + i_turn * len(agent_roles)]['content']}'"
        

    def generate(self, prompts: DataProto):
        agent_roles = prompts.meta_info["agent_roles"]
        system_prompts = prompts.meta_info["system_prompts"]
        finish_flag = prompts.meta_info["finish_flag"]
        max_num_turns = self.config.max_num_turns

        rollout_wg = self.rollout_wg_dict

        # tokenizers = {role: wg.tokenizer for role, wg in rollout_wg.items()}
        tokenizers = self.tokenizers
        for role in rollout_wg.keys():
            tokenizers[role].padding_side = "left"
            if tokenizers[role].pad_token is None:
                tokenizers[role].pad_token = tokenizers[role].eos_token

        prompts.meta_info['is_multi_turn'] = True

        questions = prompts.non_tensor_batch["question"]
        batch_size = len(questions)
        history, finish_flags, finish_reason = self._initialize_conversation_state(
            batch_size)

        # Multi-turn dialogue generation
        # this will change the history, finish_flags, finish_reason
        latest_outputs, conversation_history = self._run_multi_turn_conversation(
            prompts=prompts,
            tokenizers=tokenizers,
            max_num_turns=max_num_turns,
            agent_roles=agent_roles,
            system_prompts=system_prompts,
            finish_flag=finish_flag,
            history=history,
            finish_flags=finish_flags,
            finish_reason=finish_reason,
            response_length=self.config.response_length,
        )

        # Mark completion reasons
        # this will change the finish_reason
        if max_num_turns > 1:
            self._mark_unfinished_as_max_turns(finish_flags, finish_reason)

        last_round_responses = [{
            m["role"]: m["content"]
            for m in h[-2:]
        } for h in history]

        # extract information from history record
        num_gen_token_lst = {role: [] for role in agent_roles}
        stop_reason_lst = {role: [] for role in agent_roles}
        for h in history:
            _num_gen_tokens = {role: [] for role in agent_roles}
            _stop_reasons = {role: [] for role in agent_roles}
            for m in h:
                _num_gen_tokens[m["role"]].append(m["num_gen_tokens"])
                _stop_reasons[m["role"]].append(m["stop_reason"])
            for role in agent_roles:
                num_gen_token_lst[role].append(_num_gen_tokens[role])
                stop_reason_lst[role].append(_stop_reasons[role])

        tensor_dict = self._build_tensor_dict(
            last_round_responses,
            conversation_history,
            tokenizers,
            num_gen_token_lst,
            stop_reason_lst,
            max_num_turns,
            finish_reason,
        )

        # Prepare return results
        final_output = self._prepare_final_output(
            tensor_dict=tensor_dict,
            latest_outputs=latest_outputs,
            history=history,
            finish_reason=finish_reason,
            agent_roles=agent_roles,
            prompts=prompts,
            conversation_history=conversation_history,
        )
        
        if self.config.add_checking:
            try:
                self._checking(
                    history=history,
                    conversation_history=conversation_history,
                    agent_roles=agent_roles,
                    last_round_responses=last_round_responses,
                    tokenizer=tokenizers[agent_roles[0]],
                    tensor_dict=tensor_dict,
                    final_output=final_output,
                )
            except AssertionError as e:
                print("Error during checking:", e)
        
        return final_output
