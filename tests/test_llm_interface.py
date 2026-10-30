from types import SimpleNamespace

import cellforge.llm as llm_module


def test_openai_compatible_client_receives_configured_timeout(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(
        llm_module,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("MODEL_NAME", "fixture-model")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "17")

    result = llm_module.LLMInterface().generate("test")

    assert result["content"] == '{"ok": true}'
    assert captured["client"]["base_url"] == "https://router.example/v1"
    assert captured["client"]["timeout"] == 17.0


def test_token_usage_is_counted_across_calls(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm_module, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    start = dict(llm_module.USAGE)

    client = llm_module.LLMInterface()
    client.generate("a")
    client.chat_completion([{"role": "user", "content": "b"}])

    assert llm_module.usage_since(start) == {"calls": 2, "prompt_tokens": 22, "completion_tokens": 14}
