from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def coerce_chat_message(message: Any, *, role: str | None = None) -> dict[str, Any]:
    if isinstance(message, Mapping):
        payload = dict(message)
    elif hasattr(message, "model_dump"):
        payload = dict(message.model_dump())
    elif hasattr(message, "dict"):
        payload = dict(message.dict())
    else:
        payload = {"content": str(message)}

    if role is not None:
        payload["role"] = role
    elif "role" not in payload:
        payload["role"] = "user"

    if "content" not in payload or payload["content"] is None:
        payload["content"] = ""

    return payload


def build_assistant_history_message(
    content: str,
    *,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    """Build an assistant history message that lets the tokenizer own reasoning rendering."""

    message = coerce_chat_message({"content": content}, role="assistant")
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return message


def normalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize parquet/Arrow chat payloads before applying HF chat templates.

    Transformers 5 treats a numpy object array of chat messages as a batch and
    may return BatchEncoding instead of token ids. Local rollout needs one
    conversation, so ndarray payloads must become a plain list of message dicts.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, Mapping):
        return [coerce_chat_message(messages)]
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    elif hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, tuple):
        messages = list(messages)
    if isinstance(messages, list):
        if not messages:
            return []
        if all(isinstance(message, str) for message in messages):
            return [{"role": "user", "content": "\n".join(messages)}]
        return [coerce_chat_message(message) for message in messages]
    return [{"role": "user", "content": str(messages)}]


def extract_chat_template_token_ids(encoded: Any) -> list[int]:
    """Extract a flat list of token ids from HF chat-template outputs."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if isinstance(encoded, np.ndarray):
        encoded = encoded.tolist()
    elif hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple, np.ndarray)):
        if len(encoded) != 1:
            raise ValueError(f"Expected one chat template sequence, got {len(encoded)}")
        encoded = encoded[0]
        if isinstance(encoded, np.ndarray):
            encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def apply_chat_template_token_ids(tokenizer: Any, messages: Any, **kwargs: Any) -> list[int]:
    """Apply a chat template and always return plain token ids."""
    kwargs["tokenize"] = True
    encoded = tokenizer.apply_chat_template(normalize_chat_messages(messages), **kwargs)
    return extract_chat_template_token_ids(encoded)
