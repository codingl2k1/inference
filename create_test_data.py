import datasets
import re
import json
import tqdm

ds = datasets.load_dataset("glaiveai/glaive-function-calling-v2", split="train")

out_ds_size = 200


class UserAssistantNotFoundError(Exception):
    pass


class FunctionCallNotFoundError(Exception):
    pass


def extract_system_and_functions(system_str):
    # Extracting the system message and functions using regex
    # system_msg_match = re.search(r"SYSTEM:\s*(.*?)(?:\s+with access to the following functions\. Use them if required -|\s*, with no access to any external functions\.)?", system_str)
    system_msg = ""
    functions_match = re.findall(r"\{.*?\}(?=\s*\{|\s*$)", system_str, re.DOTALL)

    functions = []
    for fn in functions_match:
        try:
            # Convert string representation of dictionary to actual dictionary
            fn_dict = json.loads(fn)
            functions.append(fn_dict)
        except json.JSONDecodeError:
            # In case the string is not a valid JSON, continue without adding it to the list
            continue

    return system_msg, functions


def extract_user_and_assistant(chat_str):
    user_assist_pattern = r"USER:([\s\S]*?)ASSISTANT:([\s\S]*?)(?:FUNCTION RESPONSE|USER|$)"
    match = re.search(user_assist_pattern, chat_str)
    user_content = assistant_content = None
    if match:
        user_content = match.group(1).strip()
        assistant_content = match.group(2).strip()
        user_msg = {"role": "user", "content": user_content}
    else:
        raise UserAssistantNotFoundError(
            f"No user/assistant message found in {chat_str}"
        )

    user_msg = {"role": "user", "content": user_content}
    # Check if the assistant response is a function call
    if assistant_content and "<functioncall>" in assistant_content:
        if assistant_content.startswith("<functioncall>"):
            fn_call_pattern = r"<functioncall>([\s\S]*)<\|endoftext\|>"
            extract_content = False
        else:
            # fn_call_pattern = r"([\s\S]*?)<\|endoftext\|>\S*ASSISTANT: <functioncall> ([\s\S]*)<\|endoftext\|>"
            fn_call_pattern = r"([\s\S]*?)<\|endoftext\|>\s*ASSISTANT: <functioncall> ([\s\S]*)<\|endoftext\|>"
            extract_content = True

        # Extract the function call information
        function_call_match = re.search(fn_call_pattern, assistant_content)
        # Correcting the JSON string format
        if function_call_match:
            if extract_content:
                assistant_content = function_call_match.group(1).strip()
                function_call_str = function_call_match.group(2).strip()
            else:
                assistant_content = None
                function_call_str = function_call_match.group(1).strip()
            function_call_str = function_call_str.replace("\'", "")  # Replace single quotes with double quotes
            tool_call = json.loads(function_call_str)
        else:
            raise FunctionCallNotFoundError("No function call found in assistant response")

        assistant_msg = {"role": "assistant", "content": assistant_content, "tool_calls": [tool_call]}
    else:
        # Normal assistant response
        match = re.search(r"([\s\S]*?)<\|endoftext\|>", assistant_content)
        if match:
            assistant_content = match.group(1).strip()
        else:
            raise UserAssistantNotFoundError("No assistant message found in chat string")
        assistant_msg = {"role": "assistant", "content": assistant_content}

    return user_msg, assistant_msg


modified_ds = []

for i in tqdm.trange(out_ds_size):
    system_msg, functions = extract_system_and_functions(ds["system"][i])
    # We skip empty function calls.
    if not functions:
        continue

    # Some functions may not have parameters. Fix them
    for fn in functions:
        if not fn["parameters"]:
            fn["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

    if not system_msg:
        system_msg = "You are a helpful assistant."
    user_msg, assistant_msg = extract_user_and_assistant(ds["chat"][i])

    modified_ds.append({
        "system": system_msg,
        "functions": functions,
        "user": user_msg,
        "assistant": assistant_msg,
    })

    if len(modified_ds) == 100:
        break

# Write into a jsonl file readable by datasets
with open("test_data.jsonl", "w") as f:
    for item in modified_ds:
        f.write(json.dumps(item) + "\n")

