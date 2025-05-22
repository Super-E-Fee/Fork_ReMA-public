# ReMA Trainer

This document explains the multi-turn dialogue generation functionality in ReMA, particularly focusing on the `multi_turn_generate_sequences` method's data structures and processing flow.

## Multi-Turn Dialogue Generation

The `multi_turn_generate_sequences` method, found in `vllm_rollout_spmd.py`, handles multi-turn dialogue generation with multiple agent roles.

### Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| prompts | DataProto | Contains non_tensor_batch["question"] with user questions (batch_size) |
| tokenizer | PreTrainedTokenizer | Tokenizer for processing text |
| max_num_turns | int | Maximum number of conversation turns |
| agent_roles | List[str] | List of agent roles that participate in the conversation |
| finish_flag | str | String marker indicating conversation completion |
| system_prompts | Dict[str, str] | System prompts for each role |

### Data Flow and Processing

1. **Initialization**:
   - `history`: List of conversation history for each batch item `[batch_size]`
   - `finish_flags`: Boolean array of shape `[batch_size]` indicating completed conversations
   - `finish_reason`: List of completion reasons `[batch_size]`

2. **Conversation Execution**:
   - For each turn (up to `max_num_turns`):
     - For each role in `agent_roles`:
       - Prepare prompts using chat history
       - Generate responses
       - Update history and check for conversation completion

3. **Tensor Dictionary Construction**:
   - For each role:
     - `input_ids`: Padded tensor of shape `[batch_size, max_length]`
     - `labels`: Padded tensor of shape `[batch_size, max_length]` (with -100 for ignored positions)
     - `step_ids`: Padded tensor of shape `[batch_size, max_length]` (with -100 for ignored positions)
     - `attention_mask`: Binary tensor of shape `[batch_size, max_length]`
     - `position_ids`: Tensor of shape `[batch_size, max_length]` computed from attention mask

### Return Structure

The function returns a `DataProto` object containing:

#### Tensor Batch
For each role in `agent_roles`, the following tensors are included with prefix `{role}_`:
- `{role}_input_ids`: [batch_size, max_length]
- `{role}_labels`: [batch_size, max_length]
- `{role}_step_ids`: [batch_size, max_length]
- `{role}_attention_mask`: [batch_size, max_length]
- `{role}_position_ids`: [batch_size, max_length]

#### Non-Tensor Batch
- `question`: Original questions [batch_size]
- `finish_reason`: Completion reason for each conversation [batch_size]
- `num_turns`: Number of turns for each conversation [batch_size]
- `response`: Final model responses [batch_size]
- `history`: Padded conversation history [batch_size, 2*max_num_turns]
- `{role}_conversation_history`: Role-specific conversation history [batch_size, 2*max_num_turns]

### Encoding Process

The `encode_conversation` function is used to convert conversations into model inputs:
1. Processes each message in conversation
2. For assistant messages:
   - Encodes query and response
   - Creates `input_ids` combining query and response
   - Creates `labels` with IGNORE_INDEX (-100) for non-target positions
   - Generates `step_ids` tracking generation steps

### Special Handling

- Different roles have different prompt structures
- First role receives the question directly, subsequent roles receive the question and first role's instruction
- Padding is applied to ensure uniform tensor shapes across the batch
- Position IDs are computed from attention masks 