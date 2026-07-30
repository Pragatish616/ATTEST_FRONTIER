"""LLM provider router.

Single entrypoint for every LLM call in the codebase: `complete()`. Anthropic
is the primary provider; Groq and Gemini are tried in order as fallbacks on
error or rate limit (PLAN.md §6). Every call is logged with model, tokens,
latency, and cost (CLAUDE.md hard rule) — callers persist that into the
`verifications` table themselves; this module just returns it on LLMResult.

`temperature=0` is hardcoded for every provider call. Non-determinism in a
judge is a bug (CLAUDE.md hard rule) — there is no parameter to override it.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Literal

import structlog
from pydantic import BaseModel, ValidationError

from attest.config import settings

logger = structlog.get_logger(__name__)

ModelTier = Literal["fast", "judge"]

_TEMPERATURE = 0
_MAX_TOKENS = 4096

_REPAIR_PROMPT = (
    "Your previous response could not be parsed as valid JSON matching the "
    "required schema. Return valid JSON only: no prose, no markdown code "
    "fences, no commentary before or after. Here is your previous response "
    "to correct:\n\n{broken}"
)

# USD per 1M tokens, (input, output). Used only for cost visibility, not billing.
_PRICING: dict[tuple[str, ModelTier], tuple[float, float]] = {
    ("anthropic", "fast"): (0.80, 4.00),
    ("anthropic", "judge"): (3.00, 15.00),
    ("groq", "fast"): (0.05, 0.08),
    ("groq", "judge"): (0.59, 0.79),
    ("gemini", "fast"): (0.075, 0.30),
    ("gemini", "judge"): (1.25, 5.00),
}


class LLMResult(BaseModel):
    """Return value of `complete()`. Never raises what it can report here."""

    text: str
    parsed: BaseModel | dict | list | None = None
    provider: str
    model: str
    model_tier: ModelTier
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float


class LLMError(Exception):
    """Raised when every configured provider failed or none is configured."""


ProviderCall = Callable[[str, str], Awaitable[tuple[str, int, int]]]


async def _call_anthropic(prompt: str, model: str) -> tuple[str, int, int]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


async def _call_groq(prompt: str, model: str) -> tuple[str, int, int]:
    from groq import AsyncGroq

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=model,
        temperature=_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


async def _call_gemini(prompt: str, model: str) -> tuple[str, int, int]:
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": _TEMPERATURE},
    )
    text = response.text or ""
    usage = response.usage_metadata
    tokens_in = usage.prompt_token_count if usage else 0
    tokens_out = usage.candidates_token_count if usage else 0
    return text, tokens_in, tokens_out


_PROVIDER_CHAIN: list[tuple[str, ProviderCall]] = [
    ("anthropic", _call_anthropic),
    ("groq", _call_groq),
    ("gemini", _call_gemini),
]


# Transient upstream failures — rate limits and capacity blips — are the single
# most likely way a live demo dies, and they are recoverable by construction:
# the provider is telling us to wait, not that the request was wrong. Retrying
# the SAME provider before advancing the chain turns a fatal run into a slow
# one, which matters most when only one provider key is configured.
_TRANSIENT_MARKERS = (
    "429",
    "503",
    "resource_exhausted",
    "unavailable",
    "rate limit",
    "overloaded",
    "high demand",
    "try again later",
)
_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _suggested_delay(exc: Exception, fallback: float) -> float:
    """Honour the provider's own retry hint when it gives one, capped."""
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return min(float(match.group(1)), 30.0)
    return fallback


async def _call_with_retry(
    call: ProviderCall, prompt: str, model: str, provider: str
) -> tuple[str, int, int]:
    max_retries = settings.llm_transient_retries
    delay = settings.llm_retry_base_delay_s
    for attempt in range(max_retries + 1):
        try:
            return await call(prompt, model)
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised if not transient
            if attempt >= max_retries or not _is_transient(exc):
                raise
            wait_s = _suggested_delay(exc, delay)
            logger.warning(
                "llm_transient_retry",
                provider=provider,
                model=model,
                attempt=attempt + 1,
                wait_s=round(wait_s, 1),
                error=str(exc)[:160],
            )
            await asyncio.sleep(wait_s)
            delay *= 3
    raise AssertionError("unreachable")  # pragma: no cover


def _api_key_for(provider: str) -> str | None:
    return getattr(settings, f"{provider}_api_key", None)


def _model_for(provider: str, tier: ModelTier) -> str | None:
    return getattr(settings, f"{provider}_{tier}_model", None)


def _cost_usd(provider: str, tier: ModelTier, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = _PRICING[(provider, tier)]
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)


def _strip_json(raw: str) -> str:
    """Strip code fences and slice to the outermost JSON object/array."""
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end == -1 or end < start:
        return text
    return text[start : end + 1]


def _try_parse(text: str, schema: type[BaseModel]) -> BaseModel | None:
    for candidate in (text, _strip_json(text)):
        try:
            return schema.model_validate_json(candidate)
        except (ValidationError, ValueError, json.JSONDecodeError):
            continue
    return None


async def complete(
    prompt: str,
    *,
    model_tier: ModelTier,
    schema: type[BaseModel] | None = None,
) -> LLMResult:
    """Run one LLM completion, trying providers in order until one succeeds.

    If `schema` is given, the response is parsed as JSON into that model. On
    parse failure, a local repair (strip fences / isolate braces) is tried
    first; if that still fails, one extra call to the *same* provider is made
    with a "return valid JSON only" nudge.

    Once a provider **responds**, that response is final for this call: there
    is no fallback to the next provider on parse failure alone. The provider
    chain advances only when a provider call itself raises. A provider that
    answers but can't format JSON is unlikely to do better via a different
    model, and re-asking a different provider risks answer-content drift
    between providers for the same claim — so the caller gets a degraded
    `parsed=None` result (every verifier has a "parsed is None -> degrade"
    path) rather than a silently different second opinion. See
    CONTRACT_CHANGE_REQUEST.md for the rejected alternative.
    """
    failures: list[str] = []
    for provider, call in _PROVIDER_CHAIN:
        if not _api_key_for(provider):
            continue
        model = _model_for(provider, model_tier)
        if not model:
            continue

        start = time.perf_counter()
        try:
            text, tokens_in, tokens_out = await _call_with_retry(call, prompt, model, provider)
        except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
            logger.warning("llm_provider_failed", provider=provider, model=model, error=str(exc))
            failures.append(f"{provider}: {exc}")
            continue

        parsed: BaseModel | dict | list | None = None
        if schema is not None:
            parsed = _try_parse(text, schema)
            if parsed is None:
                try:
                    repaired_text, r_in, r_out = await call(
                        _REPAIR_PROMPT.format(broken=text), model
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("llm_repair_call_failed", provider=provider, error=str(exc))
                else:
                    tokens_in += r_in
                    tokens_out += r_out
                    parsed = _try_parse(repaired_text, schema)
                    if parsed is not None:
                        text = repaired_text
            if parsed is None:
                # Deliberate: no fallback to the next provider (see docstring).
                # Logged loudly because every downstream verifier degrades on
                # parsed=None, and that must never be invisible.
                logger.warning(
                    "llm_schema_parse_failed_after_repair",
                    provider=provider,
                    model=model,
                    model_tier=model_tier,
                )

        latency_ms = int((time.perf_counter() - start) * 1000)
        cost_usd = _cost_usd(provider, model_tier, tokens_in, tokens_out)
        logger.info(
            "llm_call",
            provider=provider,
            model=model,
            model_tier=model_tier,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            schema_parsed=parsed is not None if schema is not None else None,
        )
        return LLMResult(
            text=text,
            parsed=parsed,
            provider=provider,
            model=model,
            model_tier=model_tier,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    reason = "; ".join(failures) if failures else "no provider has an API key configured"
    raise LLMError(f"all providers failed for model_tier={model_tier}: {reason}")


__all__ = ["complete", "LLMResult", "LLMError", "ModelTier"]
