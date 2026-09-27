# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise three concurrent chats across staggered, synthetic tool returns."""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def chat(base_url, messages, tools):
    payload = {
        "model": "active",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result["choices"][0]["message"], time.perf_counter() - started


def session(base_url, index):
    record = f"R-{index}"
    value = 7301 + index * 17
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_number",
                "description": "Return the private number for a record.",
                "parameters": {
                    "type": "object",
                    "properties": {"record": {"type": "string"}},
                    "required": ["record"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": (
                f"Use lookup_number to get the private number for record {record}. "
                "Do not guess. After receiving the tool result, reply with only "
                f"'{record} = NUMBER', replacing NUMBER with the returned number."
            ),
        }
    ]
    assistant, first_seconds = chat(base_url, messages, tools)
    calls = assistant.get("tool_calls") or []
    assert len(calls) == 1, assistant
    call = calls[0]
    assert call["function"]["name"] == "lookup_number", call
    assert json.loads(call["function"]["arguments"]) == {"record": record}, call
    messages.append(assistant)
    # These are controlled fixture results, not external tool execution.
    time.sleep(1 + index * 2)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps({"record": record, "number": value}),
        }
    )
    answer, return_seconds = chat(base_url, messages, tools)
    expected = f"{record} = {value}"
    assert answer.get("content", "").strip() == expected, answer
    return {
        "session": record,
        "correct": True,
        "tool_call_seconds": first_seconds,
        "tool_return_seconds": return_seconds,
        "answer": answer["content"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(session, args.base_url, index) for index in range(3)]
        print(json.dumps([future.result() for future in futures], indent=2))
