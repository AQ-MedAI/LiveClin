# api_client.py
# -*- coding: utf-8 -*-

import os
import time
import base64
import mimetypes
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import openai
from tenacity import retry as async_retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# =============================================================================
# Configuration (open-source safe defaults)
# =============================================================================

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/v1").strip()
DEFAULT_RETRY_WAIT_SECONDS = int(os.getenv("RETRY_WAIT_SECONDS", "1"))
DEFAULT_RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))

# Safety limit for base64-encoded image payloads (data URI length)
MAX_BASE64_SIZE_BYTES = int(os.getenv("MAX_BASE64_SIZE_BYTES", str(20 * 1024 * 1024)))


RETRYABLE_EXCEPTIONS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


# =============================================================================
# Client
# =============================================================================

class CustomAPIClient:
    """
    A reusable asynchronous client for OpenAI-compatible APIs.

    Features:
    - API key + base URL configuration
    - Model selection
    - Multimodal message building (text + image URLs or local files encoded to base64)
    - Retry with tenacity
    - Conversation history management
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_base: str = DEFAULT_API_BASE_URL,
        retry_wait: int = DEFAULT_RETRY_WAIT_SECONDS,
        retry_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
        timeout_seconds: float = 80.0,
    ):
        if not model_name:
            raise ValueError("model_name cannot be empty.")

        self.api_key = api_key or ""
        self.model_name = model_name
        self.api_base = api_base
        self.retry_wait = retry_wait
        self.retry_attempts = retry_attempts
        self.timeout_seconds = timeout_seconds

        self.conversation_history: List[Dict[str, Any]] = []

        try:
            self.async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout_seconds,
                max_retries=0,  # we do our own retry via tenacity
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AsyncOpenAI client: {e}") from e

    def close(self):
        """
        Optional explicit close. OpenAI's client may not require it, but we provide it for cleanup.
        """
        # The OpenAI Python SDK doesn't always expose an explicit close on AsyncOpenAI;
        # keep this method for compatibility with cleanup calls.
        return

    # -------------------------------------------------------------------------
    # Image helpers
    # -------------------------------------------------------------------------

    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encode a local image file into a base64 data URI."""
        try:
            if not image_path.is_file():
                print(f"Warning: Image file not found: {image_path}")
                return None

            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type or not mime_type.startswith("image"):
                print(f"Warning: Unsupported or unknown image mime type for: {image_path}")
                return None

            with open(image_path, "rb") as f:
                binary = f.read()

            base64_str = base64.b64encode(binary).decode("utf-8")
            data_uri = f"data:{mime_type};base64,{base64_str}"

            if len(data_uri) > MAX_BASE64_SIZE_BYTES:
                print(
                    f"Warning: Encoded image exceeds limit ({len(data_uri)} bytes) and will be skipped: {image_path}"
                )
                return None

            return data_uri

        except Exception as e:
            print(f"Warning: Failed to encode image {image_path}: {e}")
            return None

    def _build_current_user_content(
        self, text_prompt: str, image_sources: List[Union[str, Path]]
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-style content list for a multimodal user message:
          [{"type":"text","text":...}, {"type":"image_url","image_url":{"url":...}}, ...]
        """
        content_list: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        processed = 0

        for source in image_sources:
            image_url_data: Optional[Dict[str, str]] = None

            if isinstance(source, Path):
                base64_uri = self._encode_image_to_base64(source)
                if base64_uri:
                    image_url_data = {"url": base64_uri}

            elif isinstance(source, str) and source.startswith(("http://", "https://")):
                image_url_data = {"url": source}

            else:
                print(f"Warning: Skipping invalid image source (must be Path or http(s) URL): {source}")
                continue

            if image_url_data:
                content_list.append({"type": "image_url", "image_url": image_url_data})
                processed += 1

        if image_sources and processed < len(image_sources):
            print(f"Warning: Processed {processed}/{len(image_sources)} image sources for this prompt.")

        return content_list

    # -------------------------------------------------------------------------
    # Retry helper
    # -------------------------------------------------------------------------

    def _apply_retry_decorator(self, func):
        """Apply tenacity async retry decorator dynamically."""
        return async_retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_fixed(self.retry_wait),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        )(func)

    # -------------------------------------------------------------------------
    # Chat completion call
    # -------------------------------------------------------------------------

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Centralized model-specific defaults.

        You can simplify this further by removing special-casing and using a single default.
        This is kept close to your original behavior.
        """
        m = (self.model_name or "").lower()

        # Defaults (OpenAI-compatible)
        params: Dict[str, Any] = {"temperature": 0, "max_tokens": 16384}

        if m == "o3":
            params = {"max_tokens": 100000}
        elif "doubao-1.5-vision-lite-250315" in m:
            params = {"temperature": 0, "top_p": 0.95, "max_tokens": 16384}
        elif "claude-3-5-haiku-20241022" in m:
            params = {"temperature": 0, "top_p": 0.95, "max_tokens": 8192}
        elif "gemini-2.5-flash" in m or "gemini-2.0-flash" in m or "gemini-1.5-pro" in m or "gemini-2.5-pro" in m:
            params = {"temperature": 0, "max_tokens": 65536}
        elif "gpt-5" in m:
            params = {"temperature": 1}  # some endpoints ignore max_tokens; leave open
        elif "qwen" in m:
            params = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}
        elif "gpt-4.1" in m:
            params = {"temperature": 0, "top_p": 0.95, "max_tokens": 32768}
        elif "o4-mini" in m:
            params = {"temperature": 0.6, "max_tokens": 100000}
        elif "gpt-4o" in m:
            params = {"temperature": 0.6, "max_tokens": 16384}
        elif "gpt-4.1-mini" in m or "gpt-4o-mini" in m:
            params = {"temperature": 0, "top_p": 0.95, "max_tokens": 16384}
        elif "doubao-1-5-vision-pro-250328" in m:
            params = {"temperature": 0, "top_p": 0.95, "max_tokens": 12288}

        return params

    async def _make_api_call(self, current_user_content: List[Dict[str, Any]], **kwargs) -> Optional[str]:
        """
        Make the actual API call including conversation history.
        Returns assistant text content (or None if cannot be extracted).
        """
        params = self._get_model_params()
        params.update(kwargs)

        messages_to_send = self.conversation_history + [{"role": "user", "content": current_user_content}]

        # Some providers use max_completion_tokens instead of max_tokens; keep your original fallback behavior.
        # We try to pass max_tokens where possible. If a provider rejects it, your retry wrapper will handle.
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages_to_send,
            "n": 1,
        }

        if "temperature" in params:
            request_kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            request_kwargs["top_p"] = params["top_p"]

        if "max_tokens" in params:
            request_kwargs["max_tokens"] = params["max_tokens"]

        # Special-case: if you truly need max_completion_tokens for some backends, add it here.
        # Keep compatibility with your original code path:
        if self.model_name == "o3" and "max_tokens" in params:
            request_kwargs["max_completion_tokens"] = params["max_tokens"]
            request_kwargs.pop("max_tokens", None)

        response = await self.async_client.chat.completions.create(**request_kwargs)
        return self._post_process(response)

    def _post_process(self, response: Any) -> Optional[str]:
        """Extract text content from an OpenAI-compatible chat completion response."""
        try:
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    return message.content.strip()
            print(f"Warning: Could not extract content from response: {response}")
            return None
        except Exception as e:
            print(f"Warning: Error processing API response: {e}. Response: {response}")
            return None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def send_prompt_async(
        self,
        text_prompt: str,
        image_sources: Optional[List[Union[str, Path]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a (possibly multimodal) prompt asynchronously with retry and history.

        Returns:
            {"status": "success", "content": "..."} or {"status": "error", "message": "..."}
        """
        if not text_prompt:
            return {"status": "error", "message": "Text prompt cannot be empty."}

        image_sources = image_sources or []
        current_user_content = self._build_current_user_content(text_prompt, image_sources)

        has_text = any(item.get("type") == "text" and item.get("text") for item in current_user_content)
        has_image = any(item.get("type") == "image_url" for item in current_user_content)
        if not (has_text or has_image):
            return {"status": "error", "message": "Failed to build valid content for API call (no text/images)."}

        retrying_call = self._apply_retry_decorator(self._make_api_call)

        try:
            start_time = time.time()
            api_content = await retrying_call(current_user_content, **kwargs)
            _ = time.time() - start_time

            if api_content:
                # Append to history only on success with content
                self.conversation_history.append({"role": "user", "content": current_user_content})
                self.conversation_history.append({"role": "assistant", "content": api_content})
                return {"status": "success", "content": api_content}

            return {"status": "error", "message": "Failed to extract content from API response."}

        except openai.AuthenticationError:
            # Do not leak key suffix; keep message generic
            return {"status": "error", "message": "Authentication failed. Check API key."}
        except openai.BadRequestError as e:
            msg = str(e)
            if "image" in msg.lower():
                return {"status": "error", "message": f"Bad Request (possible image issue): {e}"}
            return {"status": "error", "message": f"Bad Request: {e}"}
        except openai.RateLimitError as e:
            return {"status": "error", "message": f"Rate limit exceeded: {e}"}
        except Exception as e:
            return {"status": "error", "message": f"API call failed: {e}"}

    def clear_conversation_history(self):
        """Clear stored conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Return current conversation history."""
        return self.conversation_history


# =============================================================================
# Optional: local test routine (safe defaults, no private model paths)
# =============================================================================

async def test_multimodal_conversation_client():
    """
    Minimal self-test:
    - Uses env vars for model and API key
    - Generates a small dummy image locally (requires Pillow)
    """
    api_key = os.getenv("YOUR_API_KEY", "").strip()
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/v1").strip()
    model = os.getenv("MODEL_NAME", "").strip()

    if not model:
        print("Error: MODEL_NAME is not set. Example: export MODEL_NAME='gpt-4o'")
        return

    if not api_key:
        print("Error: YOUR_API_KEY is not set. Example: export YOUR_API_KEY='...'.")
        return

    sample_image_path = Path("./sample_test_image.png")
    if not sample_image_path.exists():
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (100, 100), color="red")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Test", fill=(0, 0, 0))
            img.save(sample_image_path)
        except ImportError:
            print("Pillow not installed. Install it or provide your own sample_test_image.png.")
            return

    # Example image URL (public)
    sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"

    client = CustomAPIClient(api_key=api_key, model_name=model, api_base=api_base_url)

    prompt1 = "Describe this image. What is the main subject?"
    result1 = await client.send_prompt_async(text_prompt=prompt1, image_sources=[sample_image_path])
    print("Turn 1:", result1["status"])
    if result1["status"] == "success":
        print(result1["content"])
    else:
        print(result1["message"])

    prompt2 = "What colors are prominent in what you just described?"
    result2 = await client.send_prompt_async(text_prompt=prompt2)
    print("\nTurn 2:", result2["status"])
    if result2["status"] == "success":
        print(result2["content"])
    else:
        print(result2["message"])

    prompt3 = "Now, what is shown in this image, and how is it different from the first one?"
    result3 = await client.send_prompt_async(text_prompt=prompt3, image_sources=[sample_image_url])
    print("\nTurn 3:", result3["status"])
    if result3["status"] == "success":
        print(result3["content"])
    else:
        print(result3["message"])

    print("\nConversation history entries:", len(client.get_conversation_history()))
    client.clear_conversation_history()
    print("Conversation history cleared.")


if __name__ == "__main__":
    asyncio.run(test_multimodal_conversation_client())
