import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    raw_str = cactus_complete(model,
        [{"role": "system", "content": (
    "You are a tool dispatcher.\n"
    "Output ONLY valid JSON with `function_calls`.\n"
    "Use ONLY the provided tools.\n"
    "arguments must be a JSON object.\n"
    "Include ALL required parameters.\n"
    "No extra text."
        )}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in (_enrich_tools(tools))
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    print(function_calls)
    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }

def _validate_calls(function_calls, tools):
    if not function_calls:
        return False

    tool_map = {t["name"]: t for t in tools}

    for call in function_calls:
        if not isinstance(call, dict):
            return False
        name = call.get("name")
        args = call.get("arguments", {})

        if name not in tool_map:
            return False
        if not isinstance(args, dict):
            return False
        tool = tool_map[name]
        required = tool.get("parameters", {}).get("required", [])
        props = tool.get("parameters", {}).get("properties", {})
        for key in required:
            if key not in args:
                return False
            val = args[key]

            if isinstance(val, (int, float)) and val < 0:
                return False
            if isinstance(val, str) and not val.strip():
                return False

            expected_type = (props.get(key, {}) or {}).get("type")
            if expected_type == "string" and not isinstance(val, str):
                return False
            if expected_type == "integer" and not isinstance(val, int):
                return False
            if expected_type == "number" and not isinstance(val, (int, float)):
                return False
    return True

def _enrich_tools(tools):
    import copy
    enriched = copy.deepcopy(tools)
    for t in enriched:
        props = t.get("parameters", {}).get("properties", {})
        if "time" in props:
            props["time"]["description"] = 'Time extracted from user message. Output a string.'        
        if "title" in props:
            props["title"]["description"] = 'Short title extracted directly from user message, e.g. "meeting", "groceries"'
        if "hour" in props:
            props["hour"]["description"] = 'Hour as positive integer 0-23, e.g. 6 for 6AM, 10 for 10AM'
        if "minute" in props:
            props["minute"]["description"] = 'Minute as integer 0-59, e.g. 0, 15, 30, 45'
        if "minutes" in props:
            props["minutes"]["description"] = 'Positive number of minutes, e.g. 5, 10, 20'
        if "recipient" in props:
            props["recipient"]["description"] = 'Name of person exactly as mentioned in user message, e.g. "Alice", "Bob", "John"'
        if "message" in props:
            props["message"]["description"] = 'Message content exactly as stated by user, e.g. "good morning", "hello", "I will be late"'
        if "query" in props:
            props["query"]["description"] = 'Name to search for exactly as mentioned in user message, e.g. "Bob", "Sarah"'
        if "location" in props:
            props["location"]["description"] = 'City name exactly as mentioned in user message, e.g. "London", "San Francisco"'
        if "song" in props:
            props["song"]["description"] = 'Song or playlist name exactly as mentioned in user message, e.g. "Bohemian Rhapsody", "jazz"'
    return enriched

def _is_garbled(function_calls, response_text, tools):
    tool_names = {t["name"] for t in tools}
    if not function_calls and len(response_text) > 50:
        return True
    for c in function_calls:
        name = str(c.get("name", ""))
        if "<escape>" in name or "：" in name or name not in tool_names:
            return True
    return False


def _words(text):
    text = (text or "").lower()
    for ch in ",.?;:!()[]{}":
        text = text.replace(ch, " ")
    return [w for w in text.split() if w]

def _matched_tools_by_contains(user_content, tools, max_tools=6):
    uw = _words(user_content)
    matches = []

    for t in tools or []:
        if not isinstance(t, dict) or "name" not in t:
            continue
        name = t["name"]
        if not isinstance(name, str):
            continue
        name_l = name.lower()

        score = 0
        for w in uw:
            if w and w in name_l:
                score += 1

        if score > 0:
            matches.append((score, t))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [t for score, t in matches[:max_tools]]




def generate_hybrid(messages, tools, confidence_threshold=0.99):
    tools = tools if isinstance(tools, list) else []
    enriched_all = _enrich_tools(tools)

    user_content = ""
    if isinstance(messages, list) and messages and isinstance(messages[-1], dict):
        c = messages[-1].get("content", "")
        user_content = c if isinstance(c, str) else ""

    matched = _matched_tools_by_contains(user_content, tools, max_tools=6)
    if len(matched) >= 2:
        cloud = generate_cloud(messages, tools) or {"function_calls": [], "total_time_ms": 0}
        cloud["source"] = "cloud (fallback)"
        return cloud
    if matched:
        name_set = set(t["name"] for t in matched if isinstance(t, dict) and isinstance(t.get("name"), str))
        enriched = [t for t in enriched_all if isinstance(t, dict) and t.get("name") in name_set]
        if not enriched:
            enriched = enriched_all
    else:
        enriched = enriched_all
    local = generate_cactus(messages, enriched) or {}
    calls = local.get("function_calls", [])
    if _validate_calls(calls, enriched):
        local["source"] = "on-device"
        return local
    if _is_garbled(calls, local.get("response", ""), tools):
        cloud = generate_cloud(messages, tools) or {"function_calls": [], "total_time_ms": 0}
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] = float(cloud.get("total_time_ms", 0)) + float(local.get("total_time_ms", 0))
        return cloud
    tool_names = [t["name"] for t in enriched]
    strict_messages = [
        {"role": "system", "content": (
            "You are a tool dispatcher. ONLY output tool calls, never text.\n"
            "NEVER ask clarifying questions.\n"
            "Extract arguments DIRECTLY from the user message as-is.\n"
            "Hours and minutes must be positive integers.\n"
            "Time values must be human-readable like '3:00 PM', never ISO format.\n"
            f"Available tools: {', '.join(tool_names)}"
        )},
        {"role": "user", "content": user_content}
    ]

    local2 = generate_cactus(strict_messages, enriched) or {}
    calls2 = local2.get("function_calls", [])

    if _validate_calls(calls2, enriched):
        local2["source"] = "on-device"
        local2["total_time_ms"] = float(local.get("total_time_ms", 0)) + float(local2.get("total_time_ms", 0))
        return local2
    cloud = generate_cloud(messages, tools) or {"function_calls": [], "total_time_ms": 0}
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] = (
        float(cloud.get("total_time_ms", 0)) +
        float(local.get("total_time_ms", 0)) +
        float(local2.get("total_time_ms", 0))
    )
    return cloud

def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)

