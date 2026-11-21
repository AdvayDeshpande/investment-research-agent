"""Thin wrapper around the Groq chat-completions API (OpenAI-compatible, supports tool calling)."""
import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM API cannot produce a usable response."""


class GroqLLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[float] = None, temperature: float = 0.1):
        api_key = api_key or config.GROQ_API_KEY
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Create a free key at https://console.groq.com/keys "
                           "and put it in your .env file.")
        from groq import Groq

        self.model = model or config.GROQ_MODEL
        self.temperature = temperature
        # timeout + max_retries guarantee we never hang on the network
        self._client = Groq(api_key=api_key, timeout=timeout or config.LLM_TIMEOUT_SECONDS, max_retries=2)

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        """Send the conversation (and available tools) to the model; return the assistant message."""
        import groq

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except groq.APITimeoutError as exc:
            raise LLMError(f"Groq API timed out: {exc}") from exc
        except groq.AuthenticationError as exc:
            raise LLMError("Groq rejected the API key - check GROQ_API_KEY in .env.") from exc
        except groq.RateLimitError as exc:
            raise LLMError("Groq rate limit reached - wait a minute or switch GROQ_MODEL.") from exc
        except groq.BadRequestError as exc:
            # e.g. 'tool_use_failed' when the model emits a malformed tool call
            raise LLMError(f"Groq rejected the request: {exc}") from exc
        except groq.APIError as exc:
            raise LLMError(f"Groq API error: {exc}") from exc

        usage = getattr(response, "usage", None)
        if usage:
            logger.debug("Tokens: prompt=%s completion=%s", usage.prompt_tokens, usage.completion_tokens)
        return response.choices[0].message
