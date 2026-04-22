from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drkernel"))

from kernel.workers.rollout.vllm_rollout.chat_template_utils import (
    apply_chat_template_token_ids,
    build_assistant_history_message,
    coerce_chat_message,
    extract_chat_template_token_ids,
    normalize_chat_messages,
)


def test_normalize_chat_messages_converts_numpy_object_array_to_message_list():
    messages = np.array([{"role": "user", "content": "hello"}], dtype=object)

    assert normalize_chat_messages(messages) == [{"role": "user", "content": "hello"}]


def test_normalize_chat_messages_wraps_string_and_joins_string_lists():
    assert normalize_chat_messages("hello") == [{"role": "user", "content": "hello"}]
    assert normalize_chat_messages(("line 1", "line 2")) == [
        {"role": "user", "content": "line 1\nline 2"}
    ]


def test_normalize_chat_messages_accepts_single_message_dict():
    assert normalize_chat_messages({"role": "system", "content": "rules"}) == [
        {"role": "system", "content": "rules"}
    ]


def test_coerce_chat_message_preserves_extra_fields_and_overrides_role():
    message = coerce_chat_message(
        {
            "role": "user",
            "content": "final answer",
            "reasoning_content": "private chain",
        },
        role="assistant",
    )

    assert message == {
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "private chain",
    }


def test_extract_chat_template_token_ids_accepts_batch_encoding_shape():
    assert extract_chat_template_token_ids({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == [1, 2, 3]
    assert extract_chat_template_token_ids({"input_ids": [[4, 5]], "attention_mask": [[1, 1]]}) == [4, 5]


def test_qwen35_chat_template_after_numpy_prompt_normalization_returns_token_ids():
    model_path = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-27B")
    if not (model_path / "tokenizer_config.json").exists():
        pytest.skip("Qwen3.5 tokenizer is not available in this environment")

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    messages = np.array([{"role": "user", "content": "hello"}], dtype=object)

    token_ids = apply_chat_template_token_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )

    assert isinstance(token_ids, list)
    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_qwen3_chat_template_uses_reasoning_content_field_for_history_rendering():
    model_path = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-14B")
    if not (model_path / "tokenizer_config.json").exists():
        pytest.skip("Qwen3 tokenizer is not available in this environment")

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    messages = [
        {"role": "user", "content": "first question"},
        build_assistant_history_message(
            "FINAL ANSWER",
            reasoning_content="private chain of thought",
        ),
        {"role": "user", "content": "feedback"},
    ]

    rendered = tokenizer.apply_chat_template(
        normalize_chat_messages(messages),
        tokenize=False,
        add_generation_prompt=True,
    )

    assert "private chain of thought" not in rendered
    assert "FINAL ANSWER" in rendered


def test_qwen3_chat_template_strips_inline_think_blocks_from_history():
    model_path = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-14B")
    if not (model_path / "tokenizer_config.json").exists():
        pytest.skip("Qwen3 tokenizer is not available in this environment")

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "<think>private chain</think>\n\nFINAL ANSWER"},
        {"role": "user", "content": "feedback"},
    ]

    rendered = tokenizer.apply_chat_template(
        normalize_chat_messages(messages),
        tokenize=False,
        add_generation_prompt=True,
    )

    assert "private chain" not in rendered
    assert "FINAL ANSWER" in rendered
