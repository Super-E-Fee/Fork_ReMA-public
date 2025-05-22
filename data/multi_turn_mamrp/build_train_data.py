from typing import Any, Dict, List
import jsonlines
from pathlib import Path
from prompt.math.mamrp import MTA_SYSTEM_PRMOPT, RA_SYSTEM_PRMOPT

WORK_DIR = Path(__file__).parent


def build_meta_thinking_messages(question_str: str, messages: List[Dict[str, str]]):
    ans = []
    # system prompt
    ans.append({"role": "system", "content": MTA_SYSTEM_PRMOPT})
    ans.append({"role": "user", "content": question_str})
    for i in range(len(messages)):
        if i % 2 == 0:
            assert messages[i]["role"] == "meta_thinking"
            ans.append({"role": "assistant", "content": messages[i]["content"]})
        else:
            assert messages[i]["role"]
            ans.append({"role": "user", "content": messages[i]["content"]})

    # remove last one, this is user
    ans.pop(-1)
    return ans


def build_reasoning_messages(question_str: str, messages: List[Dict[str, str]]):
    ans = []
    # system prompt
    ans.append({"role": "system", "content": RA_SYSTEM_PRMOPT})
    inst = messages[0]["content"]
    ans.append(
        {
            "role": "user",
            "content": "Question:\n{}\n\nInstruction:\n{}".format(question_str, inst),
        }
    )
    for i in range(1, len(messages)):
        if i % 2 == 0:
            assert messages[i]["role"] == "meta_thinking"
            ans.append({"role": "user", "content": messages[i]["content"]})
        else:
            assert messages[i]["role"]
            ans.append({"role": "assistant", "content": messages[i]["content"]})

    return ans


def translate_message_to_lmfct_format(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    # Initialize empty history list for storing previous turns
    history = []

    # First message is always system
    system = messages[0]["content"] if messages[0]["role"] == "system" else None

    # Build history from all message pairs except system
    history = []
    for i in range(1, len(messages) - 1, 2):
        if i + 1 < len(messages):
            history.append([messages[i]["content"], messages[i + 1]["content"]])

    # Get instruction and output from last conversation turn
    instruction = history[-1][0]
    output = history[-1][1]

    # Remove last turn from history since it becomes instruction/output
    if history:
        history.pop()

    return {
        "instruction": instruction,
        "input": None,  # Input is optional, leave empty
        "output": output,
        "system": system,
        "history": history,
    }


if __name__ == "__main__":
    metathinking_objs_to_write = []
    reasoning_objs_to_write = []
    with jsonlines.open(
        WORK_DIR / "cleaned/parsed_data_gpt4o_241120.jsonl", "r"
    ) as reader:
        for raw_obj in reader:
            question = raw_obj["question"]
            messages = raw_obj["messages"]
            assert len(messages) % 2 == 0, "messages length must be even"
            # Extract meta-thinking and reasoning content from messages
            meta_thinking_content = build_meta_thinking_messages(question, messages)
            reasoning_content = build_reasoning_messages(question, messages)

            metathinking_objs_to_write.append(
                translate_message_to_lmfct_format(meta_thinking_content)
            )
            reasoning_objs_to_write.append(
                translate_message_to_lmfct_format(reasoning_content)
            )
    (WORK_DIR/'processed').mkdir(parents=True, exist_ok=True)
    with jsonlines.open(WORK_DIR / "processed/meta_thinking.jsonl", "w") as writer:
        writer.write_all(metathinking_objs_to_write)
    with jsonlines.open(WORK_DIR / "processed/reasoning.jsonl", "w") as writer:
        writer.write_all(reasoning_objs_to_write)
