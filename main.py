#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client APIs, ten free-tier providers.

Serves POST /v1/chat/completions (OpenAI wire) and POST /v1/messages
(Anthropic wire), in JSON and in server-sent events, plus GET /v1/models in
the wire the caller's auth header names. Every client request is routed to
the highest-rated free model that meets the request's requirements, speaking
that provider's own wire (OpenAI, Anthropic or Gemini), with failover to the
next-best candidate when a provider refuses. The catalog built by probing
every provider's own model list is registered with the env and printed by
--list. Standard library only; no clock and no randomness reach any answer,
so a replayed trial reproduces the same behaviour byte for byte.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
ANTHROPIC_VERSION = "2023-06-01"
CREATED = 1735689600  # fixed timestamp for model lists and completions
POLL_INTERVAL_SEC = 0.1
PROBE_TIMEOUT_SEC = 5.0
CALL_TIMEOUT_SEC = 20.0
REGISTER_TIMEOUT_SEC = 5.0
WORK_TIMEOUT_SEC = 2.0
MAX_BODY_BYTES = 32 * 1024 * 1024
SSE_PIECE_CHARS = 16

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


# --------------------------------------------------------------------------
# The brief: three JSON documents inside $MAKEFUN_PROMPT_FILE.
# --------------------------------------------------------------------------

def documents_in(text):
    """Every top-level JSON object embedded in the brief's prose."""
    decoder = json.JSONDecoder()
    docs = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if is_record(value) and isinstance(value.get("schema"), str):
            docs.append(value)
            index = end
        else:
            index += 1
    return docs


class Brief:
    def __init__(self):
        self.providers = []   # directory entries, in the brief's order
        self.ratings = {}     # (provider, model) -> row
        self.manifest = []    # workload entries, in sending order
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA), None)
    ratings = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA), None)
    workload = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA), None)
    if directory is None:
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory.get("providers", []):
        if not is_record(raw) or not raw.get("id"):
            continue
        base = raw.get("path") or raw.get("base_url") or ""
        if "/p/" in base:
            base = base[base.index("/p/"):]
        brief.providers.append({
            "id": raw["id"],
            "name": raw.get("name", raw["id"]),
            "wire": raw.get("wire", "openai"),
            "auth": raw.get("auth", "none"),
            "key_env": raw.get("key_env", ""),
            "key": raw.get("key", ""),
            "path": base,
        })
    if ratings is not None:
        for raw in ratings.get("ratings", []):
            if not is_record(raw) or not raw.get("model") or not raw.get("provider"):
                continue
            brief.ratings[(raw["provider"], raw["model"])] = {
                "model": raw["model"],
                "provider": raw["provider"],
                "rating_milli": raw.get("rating_milli") or 0,
                "context_length": raw.get("context_length"),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            }
    if workload is not None:
        for raw in workload.get("manifest", []):
            if not is_record(raw) or not raw.get("id"):
                continue
            requires = raw.get("requires") if is_record(raw.get("requires")) else {}
            brief.manifest.append({
                "id": raw["id"],
                "api": raw.get("api", "openai"),
                "stream": raw.get("stream") is True,
                "requires": requires,
            })
    return brief


# --------------------------------------------------------------------------
# HTTP plumbing (client side): one helper for probes, calls, register, work.
# --------------------------------------------------------------------------

def http_request(method, url, headers=None, body=None, timeout=PROBE_TIMEOUT_SEC):
    """Returns (status, body_text); status 0 on a transport failure."""
    data = None
    final_headers = dict(headers or {})
    if body is not None:
        data = body.encode("utf-8") if isinstance(body, str) else body
        final_headers.setdefault("content-type", "application/json")
    request = urllib.request.Request(url, data=data, headers=final_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read().decode("utf-8", "replace")
        except Exception:
            return err.code, ""
    except Exception:
        return 0, ""


def auth_headers(provider):
    scheme, key = provider["auth"], provider["key"]
    if scheme == "bearer":
        return {"authorization": "Bearer %s" % key}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_base(provider):
    return ENV_URL + provider["path"]


def catalog_url(provider):
    if provider["wire"] == "gemini":
        return provider_base(provider) + "/v1beta/models"
    return provider_base(provider) + "/v1/models"


def infer_url(provider, model):
    base = provider_base(provider)
    if provider["wire"] == "openai":
        return base + "/v1/chat/completions"
    if provider["wire"] == "anthropic":
        return base + "/v1/messages"
    return base + "/v1beta/models/%s:generateContent" % urllib.parse.quote(model, safe="")


# --------------------------------------------------------------------------
# The catalog: probe every provider's own listing, in its own wire.
# --------------------------------------------------------------------------

class Probe:
    def __init__(self, provider):
        self.provider = provider
        self.ok = False
        self.models = []  # [{id, free, context_length}] as listed


def parse_listing(wire, text):
    try:
        body = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not is_record(body):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    models = []
    for item in items:
        if not is_record(item):
            continue
        if wire == "gemini":
            model_id = str(item.get("name", "")).removeprefix("models/")
            context = item.get("context_length", item.get("inputTokenLimit"))
        else:
            model_id = item.get("id", "")
            context = item.get("context_length", item.get("context_window"))
        if not model_id:
            continue
        models.append({
            "id": model_id,
            "free": item.get("free") if isinstance(item.get("free"), bool) else None,
            "context_length": context if isinstance(context, (int, float)) else None,
        })
    return models


def probe_provider(provider):
    probe = Probe(provider)
    if not ENV_URL:
        return probe
    headers = {"accept": "application/json", **auth_headers(provider)}
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, text = http_request("GET", catalog_url(provider), headers=headers,
                                timeout=PROBE_TIMEOUT_SEC)
    if status < 200 or status >= 300:
        log("probe of %s answered HTTP %d" % (provider["id"], status))
        return probe
    probe.models = parse_listing(provider["wire"], text)
    probe.ok = True
    return probe


def build_catalog(brief, probes):
    """The one catalog document: --list's output and the register body share it."""
    providers = []
    for probe in probes:
        provider = probe.provider
        listed = {}
        if probe.ok:
            for model in probe.models:
                listed[model["id"]] = model
        models = []
        ids = set(listed)
        for (row_provider, model_id), row in brief.ratings.items():
            if row_provider == provider["id"]:
                ids.add(model_id)
        for model_id in sorted(ids):
            row = brief.ratings.get((provider["id"], model_id))
            listed_model = listed.get(model_id)
            if row is not None:
                free = row["free"]
            elif listed_model is not None and listed_model["free"] is not None:
                free = listed_model["free"]
            else:
                free = False
            context = None
            if row is not None and row["context_length"] is not None:
                context = row["context_language"] if False else row["context_length"]
            elif listed_model is not None:
                context = listed_model["context_length"]
            # A model only the table knows is not in the provider's own listing:
            # the listing is the catalog's truth, so an unlisted model is only
            # carried when the probe itself failed and the table is all we have.
            if listed_model is None and probe.ok:
                continue
            models.append({"id": model_id, "free": free, "context_length": context})
        providers.append({
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"), ensure_ascii=False)


def free_model_ids(catalog):
    ids = []
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model["free"] is True and model["id"] not in ids:
                ids.append(model["id"])
    return ids


# --------------------------------------------------------------------------
# Wire parsing helpers, mirroring what each API's readers take from a body.
# --------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def client_json_mode(api, body):
    if api != "openai":
        return False
    response_format = body.get("response_format")
    if is_record(response_format):
        return response_format.get("type") in ("json_object", "json_schema")
    return False


def body_requirements(api, body):
    """Requirements read off a request body, the manifest's fallback."""
    vision = False
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": client_json_mode(api, body),
    }


def meets_requirements(row, requires):
    if requires.get("tools") is True and not row["tools"]:
        return False
    if requires.get("vision") is True and not row["vision"]:
        return False
    if requires.get("json") is True and not row["json"]:
        return False
    min_context = requires.get("min_context")
    if isinstance(min_context, (int, float)):
        if row["context_length"] is None or row["context_length"] < min_context:
            return False
    return True


# --------------------------------------------------------------------------
# Request translation between the wires.
# --------------------------------------------------------------------------

def data_url_of(media_type, data):
    return "data:%s;base64,%s" % (media_type or "image/png", data)


def parse_data_url(url):
    """(media_type, base64 data) for a data: URL, else None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    if not header.endswith(";base64"):
        return None
    return header[len("data:"):-len(";base64")] or "image/png", payload


def openai_image_to_anthropic(image_url):
    url = image_url.get("url", "") if is_record(image_url) else ""
    parsed = parse_data_url(url)
    if parsed is not None:
        media_type, data = parsed
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def anthropic_image_to_openai(source):
    if is_record(source) and source.get("type") == "base64":
        return {"type": "image_url",
                "image_url": {"url": data_url_of(source.get("media_type"), source.get("data", ""))}}
    if is_record(source) and source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return {"type": "image_url", "image_url": {"url": ""}}


def tool_result_text(content):
    if isinstance(content, str):
        return content
    return blocks_text(content)


def arguments_object(value):
    if is_record(value):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if is_record(parsed) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# --- OpenAI client body -> Anthropic provider body ------------------------

def openai_message_to_anthropic(message):
    role = message.get("role")
    content = message.get("content")
    if role == "user":
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        blocks = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    blocks.append(openai_image_to_anthropic(block.get("image_url")))
        return [{"role": "user", "content": blocks}]
    if role == "assistant":
        blocks = []
        if isinstance(content, str) and content != "":
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if is_record(block) and block.get("type") == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": call["function"].get("name", ""),
                    "input": arguments_object(call["function"].get("arguments")),
                })
        return [{"role": "assistant", "content": blocks}]
    if role == "tool":
        return [{"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": message.get("tool_call_id", ""),
            "content": content if isinstance(content, str) else tool_result_text(content),
        }]}]
    return []


def openai_to_anthropic(body, model):
    out = {"model": model}
    system_parts = []
    messages = []
    for message in body.get("messages", []):
        if not is_record(message):
            continue
        if message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if message.get("role") == "developer":
            continue  # the OpenAI reader does not count it as system either
        converted = openai_message_to_anthropic(message)
        for item in converted:
            # Merge a bare tool_result into the previous user message, so the
            # last user message's text stays the one the client sent.
            if (item["role"] == "user" and messages and messages[-1]["role"] == "user"
                    and isinstance(item["content"], list)
                    and all(b.get("type") == "tool_result" for b in item["content"])):
                previous = messages[-1]
                if isinstance(previous["content"], str):
                    previous["content"] = [{"type": "text", "text": previous["content"]}]
                previous["content"] = list(previous["content"]) + item["content"]
            else:
                messages.append(item)
    if system_parts:
        out["system"] = "\n".join(system_parts)
    out["messages"] = messages
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    elif body.get("max_completion_tokens") is not None:
        out["max_tokens"] = body["max_completion_tokens"]
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool) and is_record(tool.get("function")):
                function = tool["function"]
                tools.append({
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters") or {"type": "object"},
                })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, str):
        out["tool_choice"] = {"type": {"auto": "auto", "none": "none", "required": "any"}.get(choice, "auto")}
    elif is_record(choice) and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
    if body.get("stop") is not None:
        stop = body["stop"]
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    for field in ("temperature", "top_p"):
        if body.get(field) is not None:
            out[field] = body[field]
    return out


# --- Anthropic client body -> OpenAI provider body -------------------------

def anthropic_message_to_openai(message):
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": ""}]
    if role == "assistant":
        text_parts = []
        calls = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, separators=(",", ":")),
                    },
                })
        out = {"role": "assistant", "content": "".join(text_parts)}
        if calls:
            out["tool_calls"] = calls
        return [out]
    # user: text and image blocks form the user message; tool_results follow it
    parts = []
    tool_messages = []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            parts.append(anthropic_image_to_openai(block.get("source")))
        elif block.get("type") == "tool_result":
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": tool_result_text(block.get("content")),
            })
    out = []
    if parts:
        if len(parts) == 1 and parts[0]["type"] == "text":
            out.append({"role": "user", "content": parts[0]["text"]})
        else:
            out.append({"role": "user", "content": parts})
    out.extend(tool_messages)
    return out


def anthropic_to_openai(body, model):
    out = {"model": model}
    messages = []
    system = blocks_text(body.get("system")) if "system" in body else ""
    if system != "":
        messages.append({"role": "system", "content": system})
    for message in body.get("messages", []):
        if is_record(message):
            messages.extend(anthropic_message_to_openai(message))
    out["messages"] = messages
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool):
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema") or {"type": "object"},
                    },
                })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        kind = choice.get("type")
        if kind == "tool" and choice.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif kind in ("auto", "none"):
            out["tool_choice"] = kind
        elif kind == "any":
            out["tool_choice"] = "required"
    if body.get("stop_sequences") is not None:
        out["stop"] = list(body["stop_sequences"])
    for field in ("temperature", "top_p"):
        if body.get(field) is not None:
            out[field] = body[field]
    return out


# --- Either client body -> Gemini provider body ----------------------------

def gemini_tool_name_lookup(body, api):
    """tool_call id -> function name, from the body's assistant messages."""
    lookup = {}
    for message in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(message):
            continue
        if api == "openai" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if is_record(call) and is_record(call.get("function")):
                    lookup[call.get("id", "")] = call["function"].get("name", "")
        if api == "anthropic" and isinstance(message.get("content"), list):
            for block in message["content"]:
                if is_record(block) and block.get("type") == "tool_use":
                    lookup[block.get("id", "")] = block.get("name", "")
    return lookup


def openai_part_to_gemini(block):
    if not is_record(block):
        return None
    if block.get("type") == "text":
        return {"text": block.get("text", "")}
    if block.get("type") == "image_url":
        image_url = block.get("image_url") if is_record(block.get("image_url")) else {}
        parsed = parse_data_url(image_url.get("url", ""))
        if parsed is not None:
            media_type, data = parsed
            return {"inlineData": {"mimeType": media_type, "data": data}}
        return {"fileData": {"fileUri": image_url.get("url", "")}}
    return None


def anthropic_part_to_gemini(block):
    if not is_record(block):
        return None
    if block.get("type") == "text":
        return {"text": block.get("text", "")}
    if block.get("type") == "image":
        source = block.get("source") if is_record(block.get("source")) else {}
        if source.get("type") == "base64":
            return {"inlineData": {"mimeType": source.get("media_type", "image/png"),
                                   "data": source.get("data", "")}}
        return {"fileData": {"fileUri": source.get("url", "")}}
    return None


def to_gemini(body, api, model):
    out = {}
    lookup = gemini_tool_name_lookup(body, api)
    system_text = ""
    contents = []
    if api == "anthropic" and "system" in body:
        system_text = blocks_text(body.get("system"))
    for message in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if api == "openai" and role == "system":
            text = blocks_text(content)
            if text != "":
                system_text = text if system_text == "" else system_text + "\n" + text
            continue
        if api == "openai" and role == "developer":
            continue
        if role == "tool" and api == "openai":
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": lookup.get(message.get("tool_call_id", ""), ""),
                "response": {"output": content if isinstance(content, str) else tool_result_text(content)},
            }}]})
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if api == "openai":
                    part = openai_part_to_gemini(block)
                    if part is not None:
                        parts.append(part)
                else:
                    if block.get("type") == "tool_result":
                        parts.append({"functionResponse": {
                            "name": lookup.get(block.get("tool_use_id", ""), ""),
                            "response": {"output": tool_result_text(block.get("content"))},
                        }})
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {
                            "name": block.get("name", ""),
                            "args": block.get("input") or {},
                        }})
                    else:
                        part = anthropic_part_to_gemini(block)
                        if part is not None:
                            parts.append(part)
        if api == "openai" and role == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if is_record(call) and is_record(call.get("function")):
                    parts.append({"functionCall": {
                        "name": call["function"].get("name", ""),
                        "args": arguments_object(call["function"].get("arguments")),
                    }})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    out["contents"] = contents
    declarations = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if not is_record(tool):
                continue
            if api == "openai" and is_record(tool.get("function")):
                function = tool["function"]
                declarations.append({
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {"type": "object"},
                })
            elif api == "anthropic":
                declarations.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                })
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    stops = body.get("stop") if api == "openai" else body.get("stop_sequences")
    if stops is not None:
        config["stopSequences"] = [stops] if isinstance(stops, str) else list(stops)
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        config["topP"] = body["top_p"]
    if client_json_mode(api, body):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def build_provider_body(api, provider, model, body):
    """The request body for one provider, in its own wire, never streamed."""
    wire = provider["wire"]
    if wire == "gemini":
        return to_gemini(body, api, model)
    if wire == "openai":
        if api == "openai":
            out = dict(body)
            out["model"] = model
            out["stream"] = False
            return out
        return anthropic_to_openai(body, model)
    # anthropic wire
    if api == "anthropic":
        out = dict(body)
        out["model"] = model
        out["stream"] = False
        return out
    return openai_to_anthropic(body, model)


# --------------------------------------------------------------------------
# Provider answers, normalized to (text, input_tokens, output_tokens, stop).
# --------------------------------------------------------------------------

def extract_answer(wire, text):
    try:
        body = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not is_record(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        message = choices[0].get("message") if is_record(choices[0].get("message")) else {}
        content = message.get("content")
        if isinstance(content, str):
            answer_text = content
        elif isinstance(content, list):
            answer_text = "".join(
                block.get("text", "") for block in content
                if is_record(block) and block.get("type") == "text")
        else:
            answer_text = ""
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return (answer_text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                choices[0].get("finish_reason"))
    if wire == "anthropic":
        answer_text = "".join(
            block.get("text", "") for block in body.get("content", [])
            if is_record(block) and block.get("type") == "text") \
            if isinstance(body.get("content"), list) else ""
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return (answer_text, usage.get("input_tokens"), usage.get("output_tokens"),
                body.get("stop_reason"))
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    answer_text = "".join(part.get("text", "") for part in parts if is_record(part))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return (answer_text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"),
            candidates[0].get("finishReason"))


STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_FINISH = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def stop_family(reason):
    return STOP_FAMILIES.get(reason, 0)


# --------------------------------------------------------------------------
# The server.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = {probe.provider["id"]: probe for probe in probes}
        self.catalog = catalog
        self.manifest_index = 0
        self.answer_seq = 0
        self.lock = threading.Lock()

    # -- routing -----------------------------------------------------------

    def candidates(self, requires, json_mode, client_api):
        provider_order = {p["id"]: i for i, p in enumerate(self.brief.providers)}
        ranked = []
        for (provider_id, model_id), row in self.brief.ratings.items():
            if not row["free"]:
                continue
            if not meets_requirements(row, requires):
                continue
            probe = self.probes.get(provider_id)
            if probe is not None and probe.ok:
                listed = {m["id"]: m for m in probe.models}
                entry = listed.get(model_id)
                if entry is None or entry["free"] is not True:
                    continue
            provider = next((p for p in self.brief.providers if p["id"] == provider_id), None)
            if provider is None:
                continue
            # A JSON-mode OpenAI request must ride a wire that can carry it:
            # the Anthropic wire has no response_format and would echo it absent.
            if json_mode and provider["wire"] == "anthropic":
                continue
            ranked.append((provider, model_id, row))
        ranked.sort(key=lambda item: (-item[2]["rating_milli"],
                                      provider_order[item[0]["id"]], item[1]))
        return ranked

    def serve(self, api, body):
        """Answer one client request; returns (status, content_type, text)."""
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
            self.answer_seq += 1
            seq = self.answer_seq
        manifest = self.brief.manifest
        if index < len(manifest):
            requires = manifest[index]["requires"]
            stream = manifest[index]["stream"]
        else:
            requires = body_requirements(api, body)
            stream = body.get("stream") is True
        if body.get("stream") is True:
            stream = True
        json_mode = client_json_mode(api, body)
        last_status = 502
        for provider, model_id, _row in self.candidates(requires, json_mode, api):
            wire = provider["wire"]
            provider_body = build_provider_body(api, provider, model_id, body)
            headers = dict(auth_headers(provider))
            if wire == "anthropic":
                headers["anthropic-version"] = ANTHROPIC_VERSION
            status, text = http_request(
                "POST", infer_url(provider, model_id), headers=headers,
                body=json.dumps(provider_body, ensure_ascii=False),
                timeout=CALL_TIMEOUT_SEC)
            if status < 200 or status >= 300:
                last_status = status if status else 502
                continue
            answer = extract_answer(wire, text)
            if answer is None:
                continue
            answer_text, tokens_in, tokens_out, stop_raw = answer
            return self.render(api, stream, seq, model_id, answer_text,
                               tokens_in, tokens_out, stop_raw)
        return self.error(api, last_status, "no free model could serve this request")

    # -- rendering -----------------------------------------------------------

    def render(self, api, stream, seq, model_id, text, tokens_in, tokens_out, stop_raw):
        tokens_in = tokens_in if isinstance(tokens_in, (int, float)) else 0
        tokens_out = tokens_out if isinstance(tokens_out, (int, float)) else 0
        family = stop_family(stop_raw)
        if api == "openai":
            finish = OPENAI_FINISH[family]
            if not stream:
                payload = {
                    "id": "chatcmpl-%04d" % seq,
                    "object": "chat.completion",
                    "created": CREATED,
                    "model": model_id,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": text},
                                 "finish_reason": finish}],
                    "usage": {"prompt_tokens": tokens_in,
                              "completion_tokens": tokens_out,
                              "total_tokens": tokens_in + tokens_out},
                }
                return 200, "application/json", json.dumps(payload, ensure_ascii=False)
            return 200, "text/event-stream", self.openai_stream(seq, model_id, text,
                                                                finish, tokens_in, tokens_out)
        stop_reason = ANTHROPIC_STOP[family]
        if not stream:
            payload = {
                "id": "msg_%04d" % seq,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
            }
            return 200, "application/json", json.dumps(payload, ensure_ascii=False)
        return 200, "text/event-stream", self.anthropic_stream(seq, model_id, text,
                                                               stop_reason, tokens_in, tokens_out)

    @staticmethod
    def pieces(text):
        if text == "":
            return [""]
        return [text[i:i + SSE_PIECE_CHARS] for i in range(0, len(text), SSE_PIECE_CHARS)]

    def openai_stream(self, seq, model_id, text, finish, tokens_in, tokens_out):
        def chunk(delta, finish_reason, usage=None):
            payload = {"id": "chatcmpl-%04d" % seq, "object": "chat.completion.chunk",
                       "created": CREATED, "model": model_id,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if usage is not None:
                payload["choices"] = []
                payload["usage"] = usage
            return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False)
        frames = [chunk({"role": "assistant", "content": ""}, None)]
        for piece in self.pieces(text):
            frames.append(chunk({"content": piece}, None))
        frames.append(chunk({}, finish))
        frames.append(chunk({}, None, usage={"prompt_tokens": tokens_in,
                                             "completion_tokens": tokens_out,
                                             "total_tokens": tokens_in + tokens_out}))
        frames.append("data: [DONE]\n\n")
        return "".join(frames)

    def anthropic_stream(self, seq, model_id, text, stop_reason, tokens_in, tokens_out):
        def frame(event, payload):
            return "event: %s\ndata: %s\n\n" % (event, json.dumps(payload, ensure_ascii=False))
        frames = [frame("message_start", {
            "type": "message_start",
            "message": {"id": "msg_%04d" % seq, "type": "message", "role": "assistant",
                        "model": model_id, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": tokens_in, "output_tokens": 0}}})]
        frames.append(frame("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}))
        for piece in self.pieces(text):
            frames.append(frame("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": piece}}))
        frames.append(frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
        frames.append(frame("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": tokens_out}}))
        frames.append(frame("message_stop", {"type": "message_stop"}))
        return "".join(frames)

    def error(self, api, status, message):
        if api == "anthropic":
            payload = {"type": "error",
                       "error": {"type": "api_error", "message": message}}
        else:
            payload = {"error": {"message": message, "type": "api_error",
                                 "param": None, "code": "aggregator_error"}}
        return status, "application/json", json.dumps(payload, ensure_ascii=False)

    # -- GET /v1/models ------------------------------------------------------

    def models_payload(self, api):
        ids = free_model_ids(self.catalog)
        if api == "anthropic":
            data = [{"type": "model", "id": model_id, "display_name": model_id,
                     "created_at": "2026-01-01T00:00:00Z"} for model_id in ids]
            payload = {"data": data, "has_more": False}
            if ids:
                payload["first_id"] = ids[0]
                payload["last_id"] = ids[-1]
            return payload
        return {"object": "list",
                "data": [{"id": model_id, "object": "model", "created": CREATED,
                          "owned_by": "aggregator"} for model_id in ids]}


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, content_type, text):
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/models":
                api = "anthropic" if self.headers.get("x-api-key") is not None else "openai"
                self._send(200, "application/json",
                           json.dumps(aggregator.models_payload(api), ensure_ascii=False))
                return
            self._send(404, "application/json", json.dumps(
                {"error": {"message": "no route GET %s" % path, "type": "invalid_request_error",
                           "param": None, "code": "not_found"}}))

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            api = {"/v1/chat/completions": "openai", "/v1/messages": "anthropic"}.get(path)
            if api is None:
                self._send(404, "application/json", json.dumps(
                    {"error": {"message": "no route POST %s" % path,
                               "type": "invalid_request_error", "param": None, "code": "not_found"}}))
                return
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            if length > MAX_BODY_BYTES:
                status, content_type, text = aggregator.error(api, 413, "the request body is too large")
                self._send(status, content_type, text)
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
            if not is_record(body):
                status, content_type, text = aggregator.error(
                    api, 400, "the request body is not one JSON object")
                self._send(status, content_type, text)
                return
            status, content_type, text = aggregator.serve(api, body)
            self._send(status, content_type, text)

    return Handler


# --------------------------------------------------------------------------
# Trial lifecycle: probe, listen, register, poll /work, exit 0.
# --------------------------------------------------------------------------

def load_state():
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
    probes = [probe_provider(provider) for provider in brief.providers]
    catalog = build_catalog(brief, probes)
    return brief, probes, catalog


def register(port, catalog):
    status, text = http_request(
        "POST", ENV_URL + "/register",
        body=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                         "catalog": catalog}, ensure_ascii=False),
        timeout=REGISTER_TIMEOUT_SEC)
    if status < 200 or status >= 300:
        raise RuntimeError("POST /register answered HTTP %d: %s" % (status, text[:200]))
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        reply = {}
    if is_record(reply) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %d requests" % reply["requests"])


def poll_until_done():
    while True:
        status, text = http_request("GET", ENV_URL + "/work", timeout=WORK_TIMEOUT_SEC)
        if status == 200:
            try:
                work = json.loads(text)
            except json.JSONDecodeError:
                work = None
            if is_record(work) and work.get("done") is True:
                return
        time.sleep(POLL_INTERVAL_SEC)


def main(argv):
    brief, probes, catalog = load_state()
    if "--list" in argv:
        sys.stdout.write(catalog_json(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    aggregator = Aggregator(brief, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    port = server.server_address[1]
    attempts = 0
    while port == 8080 and attempts < 5:
        server.server_close()
        server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
        port = server.server_address[1]
        attempts += 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(catalog["providers"]), len(free_model_ids(catalog))))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        register(port, catalog)
    except RuntimeError as err:
        log(str(err))
        server.shutdown()
        return 1
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    poll_until_done()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
