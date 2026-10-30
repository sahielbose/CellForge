"""One interface over the LLM providers CellForge supports.

OpenAI, DeepSeek, Llama, Qwen and any custom endpoint all speak the OpenAI chat
protocol, so they share one client. Anthropic uses its own SDK. The provider is
chosen from the model name prefix, falling back to the first configured key.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    import openai
except ImportError:
    openai = None
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)

# name -> (key variable, base URL variable, default base URL, model prefixes)
OPENAI_COMPATIBLE = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1", ("gpt", "o1", "o3", "o4")),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com", ("deepseek",)),
    "llama": ("LLAMA_API_KEY", "LLAMA_BASE_URL", "https://api.llama.com/compat/v1", ("llama",)),
    "qwen": ("QWEN_API_KEY", "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", ("qwen",)),
}


def _reasoning_model(model: str) -> bool:
    # These models reject `temperature` and expect `max_completion_tokens`.
    return bool(re.match(r"^(o\d|gpt-5)", model))


class LLMInterface:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.custom_api_url = os.getenv("CUSTOM_API_URL")
        self.custom_api_key = os.getenv("CUSTOM_API_KEY")
        self.keys = {name: os.getenv(spec[0]) for name, spec in OPENAI_COMPATIBLE.items()}

    def get_config_status(self) -> Dict[str, Any]:
        status = {f"{name}_configured": bool(key) for name, key in self.keys.items()}
        status.update(
            anthropic_configured=bool(self.anthropic_api_key),
            custom_configured=bool(self.custom_api_url and self.custom_api_key),
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return status

    def _provider_for(self, model: str) -> str:
        if model.startswith("claude") and self.anthropic_api_key:
            return "anthropic"
        for name, (_, _, _, prefixes) in OPENAI_COMPATIBLE.items():
            if model.startswith(prefixes) and self.keys[name]:
                return name
        if self.custom_api_url and self.custom_api_key:
            return "custom"
        for name in OPENAI_COMPATIBLE:
            if self.keys[name]:
                return name
        if self.anthropic_api_key:
            return "anthropic"
        raise ValueError(f"No configured LLM provider can serve model '{model}'")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        return self._complete(messages, model, temperature, max_tokens)["content"]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages, model, temperature, max_tokens)

    def _complete(self, messages, model, temperature, max_tokens) -> Dict[str, Any]:
        model = model or self.model_name
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        provider = self._provider_for(model)
        try:
            if provider == "anthropic":
                content = self._anthropic(messages, model, temperature, max_tokens)
            else:
                content = self._openai_compatible(provider, messages, model, temperature, max_tokens)
        except Exception as exc:
            logger.error("LLM call to %s (%s) failed: %s", provider, model, exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc
        return {"content": content or "", "model": model, "provider": provider}

    def _openai_compatible(self, provider, messages, model, temperature, max_tokens) -> str:
        if openai is None:
            raise RuntimeError("The openai package is required for this provider")
        if provider == "custom":
            api_key = self.custom_api_key
            base_url = re.sub(r"/chat/completions/?$", "", self.custom_api_url)
        else:
            key_var, url_var, default_url, _ = OPENAI_COMPATIBLE[provider]
            api_key, base_url = os.getenv(key_var), os.getenv(url_var, default_url)
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=self.request_timeout)
        params: Dict[str, Any] = {"model": model, "messages": messages}
        if _reasoning_model(model):
            params["max_completion_tokens"] = max_tokens
        else:
            params.update(temperature=temperature, max_tokens=max_tokens)
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content

    def _anthropic(self, messages, model, temperature, max_tokens) -> str:
        if Anthropic is None:
            raise RuntimeError("The anthropic package is required for Claude models")
        client = Anthropic(
            api_key=self.anthropic_api_key,
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            timeout=self.request_timeout,
        )
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        response = client.messages.create(
            model=model,
            system=system,
            messages=turns,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def parse_json(text: Optional[str]) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating Markdown fences and prose."""

    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [text, fenced.group(1) if fenced else ""]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Response does not contain a JSON object")
