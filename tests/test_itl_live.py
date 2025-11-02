# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/test_itl_live.py
import json
import os
import time
from typing import Any

import pytest

BASE = os.environ.get("ITL_TEST_BASE_URL")
MODEL = os.environ.get("ITL_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not (BASE and MODEL),
    reason="Set ITL_TEST_BASE_URL and ITL_TEST_MODEL to run live ITL tests.",
)

openai = pytest.importorskip("openai")
from openai import OpenAI  # noqa: E402


def _stream_once(prompt: str, continuous: bool) -> dict[str, Any]:
    client = OpenAI(base_url=BASE, api_key="fake")

    stream_opts = {"include_usage": True}
    if continuous:
        stream_opts["continuous_usage_stats"] = True

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options=stream_opts,
        temperature=0.1,
        max_completion_tokens=128,
        n=1,
    )

    start = time.perf_counter()
    last_emit = start
    last_chunk = start
    last_comp = 0
    ttft_ms = 0.0
    usage_itl: list[float] = []
    chunk_itl: list[float] = []
    completion_tokens = None

    for evt in stream:
        now = time.perf_counter()
        data = (
            evt.model_dump() if hasattr(evt, "model_dump") else json.loads(evt.json())
        )

        # content-bearing chunk timing (fallback path)
        choices = data.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            # treat any delta payload as "content-bearing"
            if delta or (delta.get("content") is not None):
                if last_chunk != start:
                    chunk_itl.append((now - last_chunk) * 1000.0)
                last_chunk = now
                if ttft_ms == 0.0:
                    ttft_ms = (now - start) * 1000.0

        # usage-based timing
        usage = data.get("usage")
        if usage is not None:
            curr = usage.get("completion_tokens")
            if isinstance(curr, int):
                completion_tokens = curr
                d = curr - last_comp
                if d > 0:
                    if ttft_ms == 0.0:
                        ttft_ms = (now - start) * 1000.0
                    if last_emit == start:
                        # first emission: credit extra tokens with 0ms each
                        if d > 1:
                            usage_itl.extend([0.0] * (d - 1))
                    else:
                        dt = (now - last_emit) * 1000.0
                        per = dt / float(d)
                        usage_itl.extend([per] * d)
                    last_emit = now
                last_comp = curr

    wall_ms = (time.perf_counter() - start) * 1000.0
    return dict(
        usage_itl=usage_itl,
        chunk_itl=chunk_itl,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        wall_ms=wall_ms,
    )


def _close_enough(a: float, b: float, pct: float = 0.35) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) <= 50.0  # coarse guard for tiny runs
    return abs(a - b) / max(a, b) <= pct


def test_continuous_on_invariants():
    out = _stream_once("Explain the water cycle in 3–4 sentences.", True)
    assert out["ttft_ms"] > 0
    # If server reports completion_tokens, len(ITL) must be tokens-1
    if out["completion_tokens"] is not None:
        assert len(out["usage_itl"]) == max(out["completion_tokens"] - 1, 0)
    # Wall time ≈ TTFT + sum(usage-ITL)
    assert _close_enough(sum(out["usage_itl"]) + out["ttft_ms"], out["wall_ms"], 0.40)


def test_continuous_off_fallback_shape():
    out = _stream_once("Three quick facts about rain.", False)
    assert out["ttft_ms"] > 0
    # Many servers emit usage only at end → USAGE-ITL near-empty
    assert len(out["usage_itl"]) <= 2
    # But chunk gaps should exist since content streamed
    assert len(out["chunk_itl"]) >= 1
