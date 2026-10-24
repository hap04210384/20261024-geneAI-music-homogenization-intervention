"""OpenAI-compatible API client for the three model families.

Keys are read from repo/.env (gitignored). Any OpenAI-compatible endpoint can
reproduce the protocol by exporting the same environment variables.

Endpoints (frozen 2026-10-22, verified reachable):
  DeepSeek : https://api.deepseek.com/v1
  DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
  Zhipu    : https://open.bigmodel.cn/api/paas/v4
"""
import json
import os
import time
import urllib.request
import urllib.error

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

ENDPOINTS = {
    "deepseek": "https://api.deepseek.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}


def load_env():
    cfg = {}
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


_CFG = load_env()

API_KEYS = {
    "deepseek": _CFG.get("DEEPSEEK_API_KEY"),
    "dashscope": _CFG.get("DASHSCOPE_API_KEY"),
    "zhipu": _CFG.get("ZHIPU_API_KEY"),
}


class APIError(Exception):
    pass


def chat(family, model, messages, temperature=1.0, max_tokens=2048,
         max_retries=5, timeout=120, seed=None, extra=None):
    """Single chat completion with retry/backoff. Returns the message dict.
    `extra` merges provider-specific body fields, e.g. Zhipu thinking budget:
    extra={"thinking": {"type": "enabled", "budget_tokens": 1000}}"""
    key = API_KEYS[family]
    if not key:
        raise APIError(f"no API key configured for {family}")
    url = ENDPOINTS[family] + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if extra:
        body.update(extra)
    data = json.dumps(body).encode("utf-8")
    delay = 2.0
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"], payload
        except urllib.error.HTTPError as e:
            code = e.code
            text = e.read().decode("utf-8", errors="replace")[:500]
            if code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise APIError(f"HTTP {code}: {text}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise APIError(f"request failed after {max_retries} tries: {e}")
    raise APIError("unreachable")
