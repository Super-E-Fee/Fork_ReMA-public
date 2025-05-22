from pathlib import Path
from argparse import ArgumentParser
import jsonlines
from prompt import FINISH_FLAG

def read_data(data_path: str) -> list:
    with jsonlines.open(data_path, "r") as reader:
        data = [line for line in reader]
    return data


def clean_single_item(item: dict) -> dict:
    # 1. clean for message: if there are consecutive "meta_thinking" or "reasoning", then merge them into one message
    messages = item["messages"]
    cleaned_messages = []
    last_role = None
    for i in range(len(messages)):
        if "meta_thinking" in messages[i].keys():
            current_role = "meta_thinking"
        elif "reasoning" in messages[i].keys():
            current_role = "reasoning"
        else:
            raise ValueError("Unknown message role: {}".format(messages[i]))
        if current_role == last_role:
            cleaned_messages[-1]["content"] += "\n" + list(messages[i].values())[0]
        else:
            cleaned_messages.append({
                "role": current_role,
                "content": list(messages[i].values())[0]
            })
        last_role = current_role

    # 2. check if the last role is "reasoning"
    if cleaned_messages[-1]["role"] != "reasoning":
        raise ValueError("The last role is not reasoning: {}".format(
            cleaned_messages[-1]["role"]))

    if "\\boxed" not in cleaned_messages[-1]["content"]:
        raise ValueError("The last message does not contain \\boxed: {}".format(
            cleaned_messages[-1]["content"]))

    # 3. check if the first role is "meta_thinking"
    if cleaned_messages[0]["role"] != "meta_thinking":
        raise ValueError("The first role is not meta_thinking: {}".format(
            cleaned_messages[0]["role"]))

    # 4. check the second last message is "meta_thinking"
    # and "[FINISH]" is correctly appended in the last message
    # and make sure there is a whitespace before '[FINISH]'
    if cleaned_messages[-2]["role"] != "meta_thinking":
        raise ValueError(
            "The second last role is not meta_thinking: {}".format(
                cleaned_messages[-2]["role"]))
    if "[FINISH]" not in cleaned_messages[-1]["content"]:
        # append '[FINISH]' to the last message
        cleaned_messages[-1]["content"] = (
            cleaned_messages[-1]["content"].rstrip() + f" {FINISH_FLAG}")
    if not cleaned_messages[-1]["content"].endswith(f" {FINISH_FLAG}"):
        # split [FINISH] and drop the last part
        cleaned_messages[-1]["content"] = (
            cleaned_messages[-1]["content"].split(FINISH_FLAG)[0].strip() +
            f" {FINISH_FLAG}")

    item["messages"] = cleaned_messages
    return item


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    data = read_data(args.data_path)
    cleaned_data = []
    valid_count = 0
    invalid_count = 0

    for item in data:
        try:
            cleaned_item = clean_single_item(item)
            cleaned_data.append(cleaned_item)
            valid_count += 1
        except Exception as e:
            print(f"Line: {item['id']}, Error processing item: {e}")
            invalid_count += 1

    print(
        f"Cleaning completed. Valid items: {valid_count}, Invalid items: {invalid_count}"
    )

    write_dir = Path(__file__).parent / "cleaned"
    write_dir.mkdir(exist_ok=True, parents=True)
    write_path = write_dir / args.data_path.split("/")[-1]
    print(f"Writing cleaned data to {write_path}")
    with jsonlines.open(write_path, "w") as writer:
        writer.write_all(cleaned_data)
