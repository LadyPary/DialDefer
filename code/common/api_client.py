"""API client for OpenAI/OpenRouter models and Anthropic Claude models.

Routing is by model name: any model starting with "claude-" (or "anthropic/claude-")
goes to the Anthropic Messages API via the official `anthropic` SDK; everything else
goes through the OpenAI-compatible client exactly as before.

Claude-specific settings (environment variables, all optional):
  ANTHROPIC_API_KEY   - API key (or use `ant auth login`)
  DIALDEFER_EFFORT       - low | medium | high | xhigh | max  (default: medium, the
                        Opus 5.5 API default; set explicitly so it is recorded)
  DIALDEFER_MAX_TOKENS   - output cap incl. thinking tokens (default: 16000). The runners'
                        --max-tokens (300) is too small once thinking is on, so the
                        larger of the two is used.
  DIALDEFER_CALL_LOG     - JSONL path where every Claude call's stop_reason, token usage
                        and request id is appended (default: results/_claude_calls.jsonl)

Notes for Claude Opus 5.5:
  * `temperature` is not accepted by the API, so it is ignored for Claude models.
  * Thinking cannot be turned off; `effort` is the only depth control.
  * Server-side refusal fallbacks are deliberately NOT enabled: a fallback would
    silently answer with a different model and contaminate per-model results.
    Refusals are logged and return an empty reply (parsed as a missing answer).
"""

import json
import os
import time
from typing import List, Tuple, Union

from openai import OpenAI

try:
    import anthropic
except ImportError:  # only needed for Claude models
    anthropic = None


class FatalAPIError(BaseException):
    """Raised for non-retryable API errors; deliberately not an `Exception` subclass
    so per-row `except Exception` handlers in the runners don't swallow it."""


def is_claude_model(model: str) -> bool:
    return bool(model) and model.split("/")[-1].startswith("claude-")


def _claude_model_id(model: str) -> str:
    return model.split("/")[-1]


def create_client(api_key: str = None, base_url: str = None, model: str = None):
    """Create an API client. Returns an Anthropic client for Claude models,
    otherwise an OpenAI client with optional custom base URL."""
    if model is not None and is_claude_model(model):
        if anthropic is None:
            raise ImportError("Install the Anthropic SDK: pip install anthropic")
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
        return anthropic.Anthropic(max_retries=8)

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

    if base_url is None:
        base_url = os.getenv("API_BASE_URL")

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _log_claude_call(record: dict) -> None:
    path = os.getenv("DIALDEFER_CALL_LOG", "results/_claude_calls.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _chat_claude(client, model, messages, max_tokens, system_message):
    effort = os.getenv("DIALDEFER_EFFORT", "medium")
    max_tokens = max(max_tokens, int(os.getenv("DIALDEFER_MAX_TOKENS", "16000")))

    kwargs = dict(
        model=_claude_model_id(model),
        max_tokens=max_tokens,
        messages=messages,
        output_config={"effort": effort},
    )
    if system_message:
        kwargs["system"] = system_message

    try:
        response = client.messages.create(**kwargs)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
            anthropic.NotFoundError, anthropic.BadRequestError) as e:
        # Non-retryable (bad key, no credits, bad model/params). The runners catch
        # `Exception` per condition and would silently write empty answers for every
        # row, so escalate to a BaseException that stops the run instead.
        raise FatalAPIError(f"Fatal Claude API error, stopping run: {e}") from e

    reply = "".join(b.text for b in response.content if b.type == "text")
    stop_details = getattr(response, "stop_details", None)
    _log_claude_call({
        "ts": time.time(),
        "request_id": response._request_id,
        "model": response.model,
        "effort": effort,
        "stop_reason": response.stop_reason,
        "stop_category": getattr(stop_details, "category", None) if stop_details else None,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    })
    if response.stop_reason in ("refusal", "max_tokens"):
        print(f"[claude] stop_reason={response.stop_reason} request_id={response._request_id}")
    return reply


def chat(
    client: Union[OpenAI, "anthropic.Anthropic"],
    model: str,
    user_message: str,
    history: List[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    system_message: str = None
) -> Tuple[str, List[dict]]:
    """Send message and get response. Returns (reply, updated_history)."""
    if history is None:
        history = []

    messages = history.copy()
    messages.append({"role": "user", "content": user_message})

    if is_claude_model(model):
        if anthropic is None or not isinstance(client, anthropic.Anthropic):
            raise TypeError("Claude model requested but client is not an Anthropic client; "
                            "call create_client(model=...)")
        reply = _chat_claude(client, model, messages, max_tokens, system_message)
    else:
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        reply = response.choices[0].message.content

    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ]

    return reply, updated_history
