"""Async API client for OpenAI-compatible endpoints."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from . import EvalConfig

MAX_BASE64_BYTES = 20 * 1024 * 1024


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient errors: timeouts, connection issues, rate limits, 5xx."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


class APIClient:
    """Async client wrapping a shared AsyncOpenAI connection with per-instance history."""

    _shared_openai: Optional[openai.AsyncOpenAI] = None
    _shared_config_key: Optional[str] = None

    def __init__(self, config: EvalConfig):
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.retry_attempts = config.max_retries
        self.retry_wait = config.retry_delay

        config_key = f"{config.api_base}|{config.api_key}|{config.timeout}"
        if APIClient._shared_openai is None or APIClient._shared_config_key != config_key:
            APIClient._shared_openai = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
                timeout=config.timeout,
                max_retries=0,
            )
            APIClient._shared_config_key = config_key

        self.async_client = APIClient._shared_openai
        self.history: List[Dict[str, Any]] = []

    def clear_history(self) -> None:
        self.history = []

    def fork(self) -> "APIClient":
        """Create a lightweight copy sharing the same connection but with fresh history."""
        clone = object.__new__(APIClient)
        clone.model = self.model
        clone.temperature = self.temperature
        clone.max_tokens = self.max_tokens
        clone.retry_attempts = self.retry_attempts
        clone.retry_wait = self.retry_wait
        clone.async_client = self.async_client
        clone.history = []
        return clone

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(path: Path) -> Optional[str]:
        if not path.is_file():
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image"):
            return None
        data = base64.b64encode(path.read_bytes()).decode()
        uri = f"data:{mime};base64,{data}"
        if len(uri) > MAX_BASE64_BYTES:
            return None
        return uri

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_content(
        text: str, images: List[Union[str, Path]]
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for src in images:
            url: Optional[str] = None
            if isinstance(src, Path):
                url = APIClient._encode_image(src)
            elif isinstance(src, str) and src.startswith(("http://", "https://")):
                url = src
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    # ------------------------------------------------------------------
    # API call with retry
    # ------------------------------------------------------------------

    async def _call(
        self, content: List[Dict[str, Any]]
    ) -> Optional[str]:
        messages = self.history + [{"role": "user", "content": content}]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = await self.async_client.chat.completions.create(**kwargs)
        if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        text: str,
        images: Optional[List[Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt (with optional images) and return the result.

        Returns {"status": "success", "content": "..."} or
                {"status": "error",   "message": "..."}.
        """
        images = images or []
        content = self._build_content(text, images)

        retrying_call = retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_wait, min=1, max=60),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )(self._call)

        try:
            result = await retrying_call(content)
            if result:
                self.history.append({"role": "user", "content": content})
                self.history.append({"role": "assistant", "content": result})
                return {"status": "success", "content": result}
            return {"status": "error", "message": "Empty response from API."}
        except openai.AuthenticationError:
            return {"status": "error", "message": "Authentication failed. Check API key."}
        except openai.BadRequestError as e:
            return {"status": "error", "message": f"Bad request: {e}"}
        except openai.RateLimitError as e:
            return {"status": "error", "message": f"Rate limit: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"API error: {e}"}
