"""Language model providers.

PyraGuard never depends on a hosted model being reachable. The default
``extractive`` mode composes the response plan directly from retrieved
guidance and needs no provider at all. When a provider is configured its
output is treated as untrusted text and passes through the same grounding
checks before anyone sees it.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from abc import ABC, abstractmethod

import cv2
import numpy as np

from pyraguard.config import LLMConfig
from pyraguard.logging_utils import get_logger

log = get_logger(__name__)


def _jpeg_base64(frame: np.ndarray, max_side: int = 768) -> str:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class LLMProvider(ABC):
    name = "provider"
    supports_images = False

    @abstractmethod
    def complete(self, system: str, user: str, image: np.ndarray | None = None) -> str:
        """Return the model's text reply."""


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_images = True
    default_model = "claude-sonnet-5-5"

    def __init__(self, cfg: LLMConfig) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.cfg = cfg
        self.model = cfg.model or self.default_model
        self.client = anthropic.Anthropic(timeout=cfg.timeout_seconds)

    def complete(self, system: str, user: str, image: np.ndarray | None = None) -> str:
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_base64(image)}})
        content.append({"type": "text", "text": user})
        reply = self.client.messages.create(
            model=self.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in reply.content if getattr(block, "type", "") == "text")


class OpenAIProvider(LLMProvider):
    name = "openai"
    supports_images = True

    def __init__(self, cfg: LLMConfig) -> None:
        from openai import OpenAI

        if not cfg.model:
            raise RuntimeError("Set PYRAGUARD_LLM_MODEL to the OpenAI model you want to use")
        self.cfg = cfg
        self.client = OpenAI(timeout=cfg.timeout_seconds)

    def complete(self, system: str, user: str, image: np.ndarray | None = None) -> str:
        content: list[dict] = [{"type": "text", "text": user}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_base64(image)}"}})
        reply = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        )
        return reply.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    """Local models served by Ollama, for sites where footage may not leave the network."""

    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.model:
            raise RuntimeError("Set PYRAGUARD_LLM_MODEL to the Ollama model you want to use")
        self.cfg = cfg
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def complete(self, system: str, user: str, image: np.ndarray | None = None) -> str:
        body = {
            "model": self.cfg.model,
            "stream": False,
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.cfg.timeout_seconds) as response:  # noqa: S310 - operator configured host
            return json.loads(response.read().decode("utf-8")).get("message", {}).get("content", "")


def build_provider(cfg: LLMConfig) -> LLMProvider | None:
    """Return a provider, or ``None`` for extractive mode or when the provider cannot be created."""
    choice = cfg.provider.lower()
    if choice in ("", "extractive", "none"):
        return None
    try:
        if choice == "anthropic":
            return AnthropicProvider(cfg)
        if choice == "openai":
            return OpenAIProvider(cfg)
        if choice == "ollama":
            return OllamaProvider(cfg)
    except Exception as exc:
        log.warning("LLM provider %s unavailable (%s). Falling back to extractive generation.", choice, exc)
        return None
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
