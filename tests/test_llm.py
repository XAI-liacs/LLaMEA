import copy
import datetime as _dt
import pickle
import random
from unittest.mock import MagicMock, patch

import pytest

try:  # openai>=3 uses httpx2; fall back to httpx for openai<3
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

import llamea.llm as llm_mod  # the module that defines query
from llamea import (
    LLM,
    Anthropic_LLM,
    Dummy_LLM,
    Gemini_LLM,
    Multi_LLM,
    Ollama_LLM,
    OpenAI_LLM,
    Multi_LLM,
    LMStudio_LLM,
    MLX_LM_LLM
)


class _DummyOpenAI:
    """Stand-in that just records the kwargs used to build it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_openai(monkeypatch):
    """
    Helper that swaps out openai.OpenAI with _DummyOpenAI inside the
    already-imported iohblade.llm module.
    """
    monkeypatch.setattr(llm_mod.openai, "OpenAI", _DummyOpenAI)


def test_openai_llm_getstate_strips_client(monkeypatch):
    _patch_openai(monkeypatch)

    llm = OpenAI_LLM(api_key="sk-test", model="gpt-4-turbo")
    state = llm.__getstate__()

    assert "client" not in state
    # sanity-check something else is still there
    assert state["model"] == "gpt-4-turbo"


def test_openai_llm_deepcopy_restores_client(monkeypatch):
    _patch_openai(monkeypatch)

    original = OpenAI_LLM(api_key="sk-test", model="gpt-4o", temperature=0.17)
    clone = copy.deepcopy(original)

    # new object, equal public state
    assert clone is not original
    assert clone.model == original.model
    assert clone.temperature == original.temperature

    # brand-new client object of the dummy type
    assert isinstance(clone.client, _DummyOpenAI)
    assert clone.client is not original.client
    assert clone.client.kwargs["api_key"] == "sk-test"

    # changing the clone does not leak back
    clone.temperature = 0.99
    assert original.temperature != clone.temperature


def test_openai_llm_pickle_roundtrip(monkeypatch):
    _patch_openai(monkeypatch)

    llm = OpenAI_LLM(api_key="sk-test", model="gpt-3.5-turbo")
    blob = pickle.dumps(llm)
    revived = pickle.loads(blob)

    # revived instance has equivalent state and a fresh client
    assert revived.model == llm.model
    assert isinstance(revived.client, _DummyOpenAI)
    assert revived.client.kwargs["api_key"] == "sk-test"


class _DummyAnthropicClient:
    """Stand-in for anthropic.Anthropic that records its kwargs and requests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = MagicMock()
        self.beta = MagicMock()


def _patch_anthropic(monkeypatch):
    """
    Swaps the anthropic module inside llamea.llm for a fake, so the tests
    run without the anthropic package or network access.
    """
    fake_module = MagicMock()
    fake_module.Anthropic = _DummyAnthropicClient
    monkeypatch.setattr(llm_mod, "anthropic", fake_module)


def _anthropic_response(*texts, stop_reason="end_turn"):
    """Builds a Messages API response with one text block per string."""
    return MagicMock(
        content=[MagicMock(type="text", text=t) for t in texts],
        stop_reason=stop_reason,
    )


def test_anthropic_llm_init(monkeypatch):
    _patch_anthropic(monkeypatch)

    llm = Anthropic_LLM(api_key="sk-ant-test", model="claude-haiku-5-5")

    assert llm.model == "claude-haiku-5-5"
    assert llm.client.kwargs == {"api_key": "sk-ant-test", "max_retries": 5}


def test_anthropic_llm_query_sends_request_and_joins_text(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test", model="claude-opus-5-5", max_tokens=123)
    thinking = MagicMock(type="thinking")
    response = _anthropic_response("Hello ", "world")
    response.content = [thinking] + response.content
    llm.client.messages.create.return_value = response

    out = llm.query([{"role": "user", "content": "hi"}])

    assert out == "Hello world"  # non-text blocks are skipped
    llm.client.messages.create.assert_called_once_with(
        model="claude-opus-5-5",
        max_tokens=123,
        messages=[{"role": "user", "content": "hi"}],
    )


def test_anthropic_llm_query_moves_system_messages(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test")
    llm.client.messages.create.return_value = _anthropic_response("ok")

    llm.query(
        [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Write code."},
        ]
    )

    kwargs = llm.client.messages.create.call_args.kwargs
    assert kwargs["system"] == "You are an expert."
    assert kwargs["messages"] == [{"role": "user", "content": "Write code."}]


def test_anthropic_llm_never_sends_temperature(monkeypatch):
    # API version 2026-10-21 rejects sampling parameters on every model.
    _patch_anthropic(monkeypatch)
    with pytest.warns(DeprecationWarning):
        llm = Anthropic_LLM(api_key="sk-ant-test", temperature=0.3)
    llm.client.messages.create.return_value = _anthropic_response("ok")

    llm.query([{"role": "user", "content": "hi"}])

    kwargs = llm.client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "output_config" not in kwargs
    assert kwargs["max_tokens"] == 16000


def test_anthropic_llm_sends_effort(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test", effort="high")
    llm.client.messages.create.return_value = _anthropic_response("ok")

    llm.query([{"role": "user", "content": "hi"}])

    kwargs = llm.client.messages.create.call_args.kwargs
    assert kwargs["output_config"] == {"effort": "high"}


def test_anthropic_llm_fallbacks_use_beta_endpoint(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test", fallbacks="default")
    llm.client.beta.messages.create.return_value = _anthropic_response("OK")

    assert llm.query([{"role": "user", "content": "hi"}]) == "OK"

    kwargs = llm.client.beta.messages.create.call_args.kwargs
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    llm.client.messages.create.assert_not_called()


def test_anthropic_llm_refusal_returns_no_text(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test")
    llm.client.messages.create.return_value = _anthropic_response(
        "partial", stop_reason="refusal"
    )

    assert llm.query([{"role": "user", "content": "hi"}]) == ""


def test_anthropic_llm_unpickles_state_without_new_fields(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test")
    state = llm.__getstate__()
    del state["effort"], state["fallbacks"]

    revived = Anthropic_LLM.__new__(Anthropic_LLM)
    revived.__setstate__(state)

    assert revived.effort is None and revived.fallbacks is None
    assert isinstance(revived.client, _DummyAnthropicClient)


def test_anthropic_llm_sample_solution(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test")
    llm.client.messages.create.return_value = _anthropic_response(
        "# Description: Random search\n```python\nclass RS:\n    pass\n```"
    )

    sol = llm.sample_solution([{"role": "user", "content": "go"}])

    assert sol.name == "RS"
    assert sol.description == "Random search"
    assert "class RS" in sol.code


def test_anthropic_llm_deepcopy_and_pickle_restore_client(monkeypatch):
    _patch_anthropic(monkeypatch)
    llm = Anthropic_LLM(api_key="sk-ant-test", model="claude-sonnet-5-5")

    assert "client" not in llm.__getstate__()

    clone = copy.deepcopy(llm)
    assert clone.model == llm.model
    assert isinstance(clone.client, _DummyAnthropicClient)
    assert clone.client is not llm.client
    assert clone.client.kwargs["api_key"] == "sk-ant-test"

    revived = pickle.loads(pickle.dumps(llm))
    assert revived.max_tokens == llm.max_tokens
    assert isinstance(revived.client, _DummyAnthropicClient)
    assert revived.client.kwargs["api_key"] == "sk-ant-test"


def test_llm_instantiation():
    # Since LLM is abstract, we'll instantiate a child class
    class DummyLLM(LLM):
        def query(self, session: list):
            return "Mock response"

    llm = DummyLLM(api_key="fake", model="fake")
    assert llm.api_key == "fake"
    assert llm.model == "fake"


def test_llm_sample_solution_no_code_raises_exception():
    class DummyLLM(LLM):
        def query(self, session: list):
            return "This has no code block"

    llm = DummyLLM(api_key="x", model="y")
    with pytest.raises(
        Exception
    ):  # uses the fallback `raise Exception("Could not extract...")`
        exec(llm.sample_solution([{"role": "client", "content": "test"}]), {}, {})


def test_llm_sample_solution_good_code():
    class DummyLLM(LLM):
        def query(self, session: list):
            return "# Description: MyAlgo\n```python\nclass MyAlgo:\n  pass\n```"

    llm = DummyLLM(api_key="x", model="y")
    sol = llm.sample_solution([{"role": "client", "content": "test"}])
    assert sol.name == "MyAlgo"
    assert "class MyAlgo" in sol.code

def test_llm_sample_solution_func():
    class DummyLLM(LLM):
        def query(self, session: list):
            return "# Description: MyAlgo\n```python\ndef MyAlgo(parm1, param2):\n  pass\n```"

    llm = DummyLLM(api_key="x", model="y")
    sol = llm.sample_solution([{"role": "client", "content": "test"}])
    assert sol.name == "MyAlgo"
    assert "MyAlgo" in sol.code


def test_extract_algorithm_code_strips_main_block():
    class DummyLLM(LLM):
        def query(self, session: list):  # pragma: no cover - helper for direct method call
            return ""

    llm = DummyLLM(api_key="x", model="y")
    message = (
        "```python\n"
        "class Foo:\n"
        "    pass\n\n"
        "if __name__ == '__main__':\n"
        "    print('hi')\n"
        "```"
    )

    code = llm.extract_algorithm_code(message)
    assert "if __name__" not in code
    assert "print('hi')" not in code


def test_openai_llm_init():
    # We won't actually call OpenAI's API. Just ensure it can be constructed.
    llm = OpenAI_LLM(api_key="fake_key", model="gpt-3.5-turbo")
    assert llm.model == "gpt-3.5-turbo"


def test_ollama_llm_init():
    llm = Ollama_LLM(model="llama2.0")
    assert llm.model == "llama2.0"


def test_gemini_llm_init():
    llm = Gemini_LLM(api_key="some_key", model="gemini-2.0-flash")
    assert llm.model == "gemini-2.0-flash"


def test_gemini_llm_pins_v1beta(monkeypatch):
    """google-genai>=3 defaults to the v1 API; LLaMEA keeps v1beta."""
    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    llm = Gemini_LLM(api_key="some_key")
    assert llm.client._api_client._http_options.api_version == "v1beta"
    clone = copy.deepcopy(llm)
    assert clone.client._api_client._http_options.api_version == "v1beta"


def test_gemini_llm_query_builds_gemini_history():
    """Roles map to Gemini's user/model, parts are {"text": ...} and system
    messages become the system_instruction."""
    llm = Gemini_LLM(api_key="fake", model="gemini-test")
    chat = MagicMock()
    chat.send_message.return_value = type("R", (), {"text": "OK"})
    llm.client = MagicMock()
    llm.client.chats.create.return_value = chat

    reply = llm.query(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )

    assert reply == "OK"
    kwargs = llm.client.chats.create.call_args.kwargs
    assert kwargs["history"] == [
        {"role": "user", "parts": [{"text": "a"}]},
        {"role": "model", "parts": [{"text": "b"}]},
    ]
    assert kwargs["config"]["system_instruction"] == "be brief"
    chat.send_message.assert_called_once_with("c")


def test_gemini_llm_query_history_validates_with_sdk():
    """The history format must be accepted by google-genai itself."""
    from google.genai import types

    llm = Gemini_LLM(api_key="fake", model="gemini-test")
    llm.client = MagicMock()
    llm.client.chats.create.return_value.send_message.return_value = type(
        "R", (), {"text": "OK"}
    )
    llm.query(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    history = llm.client.chats.create.call_args.kwargs["history"]
    for item in history:
        types.Content.model_validate(item)


def test_gemini_llm_respects_api_version_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v1")
    llm = Gemini_LLM(api_key="some_key")
    assert llm.client._api_client._http_options.api_version == "v1"


def test_multi_llm_selects_random_model(monkeypatch):
    class LLMA(LLM):
        def query(self, session):
            return "# Description: A\n```python\nclass A:\n    pass\n```"

    class LLMB(LLM):
        def query(self, session):
            return "# Description: B\n```python\nclass B:\n    pass\n```"

    a = LLMA(api_key="a", model="model-a")
    b = LLMB(api_key="b", model="model-b")
    combo = Multi_LLM([a, b])

    monkeypatch.setattr(random, "choice", lambda seq: seq[1])

    sol = combo.sample_solution([{"role": "client", "content": "test"}])
    assert sol.name == "B"


def _resource_exhausted(delay_secs: int = 2) -> Exception:
    """
    Build a faux `ResourceExhausted`-style exception carrying a `retry_delay`
    attr that the retry logic recognises.
    """
    err = Exception("429 ResourceExhausted")
    err.retry_delay = _dt.timedelta(seconds=delay_secs)
    return err


def _openai_rate_limit(retry_after: int = 2) -> Exception:
    response = httpx.Response(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        request=httpx.Request("POST", "http://test"),
    )
    return llm_mod.openai.RateLimitError("quota", response=response, body=None)


def _ollama_response_error(status: int = 429) -> Exception:
    return llm_mod.ollama.ResponseError("quota", status_code=status)


def test_gemini_llm_retries_then_succeeds(monkeypatch):
    """query should sleep, retry once, then return the model reply."""
    llm = Gemini_LLM(api_key="fake", model="gemini-test")

    # -- stub out time.sleep so the test is instant
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    # First start_chat → chat.send_message raises; second returns text
    chat_fail = MagicMock()
    chat_fail.send_message.side_effect = _resource_exhausted(2)

    chat_ok = MagicMock()
    chat_ok.send_message.return_value = type("R", (), {"text": "OK-DONE"})

    fake_client = MagicMock()
    fake_client.chats.create.side_effect = [chat_fail, chat_ok]
    llm.client = fake_client

    reply = llm.query([{"role": "user", "content": "hello"}], max_retries=3)

    assert reply == "OK-DONE"
    assert fake_client.chats.create.call_count == 2  # 1 failure + 1 success
    slept.assert_called_once_with(3)  # 2 s + 1 s safety buffer


def test_gemini_llm_gives_up_after_max_retries(monkeypatch):
    """query should bubble the error once max_retries is exceeded."""
    llm = Gemini_LLM(api_key="fake", model="gemini-test")

    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    chat_fail = MagicMock()
    chat_fail.send_message.side_effect = _resource_exhausted(1)

    fake_client = MagicMock()
    fake_client.chats.create.return_value = chat_fail
    llm.client = fake_client

    with pytest.raises(Exception):
        llm.query([{"role": "user", "content": "boom"}], max_retries=2)

    # It sleeps exactly `max_retries` times (raises on the next attempt)
    assert slept.call_count == 2


def test_openai_llm_retries_then_succeeds(monkeypatch):
    llm = OpenAI_LLM(api_key="fake", model="gpt-test")

    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="DONE"))]
    llm.client.chat.completions.create = MagicMock(
        side_effect=[_openai_rate_limit(2), ok]
    )

    reply = llm.query([{"role": "user", "content": "hi"}], max_retries=2)
    assert reply == "DONE"
    assert llm.client.chat.completions.create.call_count == 2
    slept.assert_called_once_with(2)


def test_openai_llm_gives_up(monkeypatch):
    llm = OpenAI_LLM(api_key="fake", model="gpt-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    llm.client.chat.completions.create = MagicMock(
        side_effect=[_openai_rate_limit(1), _openai_rate_limit(1)]
    )

    with pytest.raises(llm_mod.openai.RateLimitError):
        llm.query([{"role": "user", "content": "boom"}], max_retries=1)
    slept.assert_called_once_with(1)


def test_ollama_llm_retries_then_succeeds(monkeypatch):
    llm = Ollama_LLM(model="llama-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    monkeypatch.setattr(
        llm_mod.ollama,
        "chat",
        MagicMock(
            side_effect=[_ollama_response_error(429), {"message": {"content": "OK"}}]
        ),
    )

    reply = llm.query([{"role": "u", "content": "hi"}], max_retries=2)
    assert reply == "OK"
    llm_mod.ollama.chat.assert_called_with(
        model=llm.model,
        messages=[{"role": "user", "content": "hi\n"}],
    )
    slept.assert_called_once_with(10)


def test_ollama_llm_gives_up(monkeypatch):
    llm = Ollama_LLM(model="llama-test")
    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)
    monkeypatch.setattr(
        llm_mod.ollama,
        "chat",
        MagicMock(side_effect=[_ollama_response_error(), _ollama_response_error()]),
    )

    with pytest.raises(llm_mod.ollama.ResponseError):
        llm.query([{"role": "u", "content": "boom"}], max_retries=1)
    slept.assert_called_once_with(10)


def test_dummy_llm():
    llm = Dummy_LLM(model="dummy-model")
    assert llm.model == "dummy-model"
    response = llm.query([{"role": "user", "content": "test"}])
    assert (
        len(response) == 921
    ), "Dummy_LLM should return a 946-character string, returned length: {}".format(
        len(response)
    )


def test_multi_llm_logger_propagates():
    class LLMA(LLM):
        def query(self, session):
            return "A"

    class LLMB(LLM):
        def query(self, session):
            return "B"

    combo = Multi_LLM([LLMA(api_key="a", model="ma"), LLMB(api_key="b", model="mb")])

    logger = MagicMock()
    combo.set_logger(logger)

    assert combo.logger is logger
    assert combo.llms[0].logger is logger
    assert combo.llms[1].logger is logger
    assert combo.llms[0].log and combo.llms[1].log
    assert combo.model == "multi-llm"

@pytest.fixture
def fake_mlx_load(monkeypatch):
    llm = object()
    tokenizer = object()
    calls = []

    def fake_load(model, model_config=None):
        calls.append((model, model_config))
        print(calls)
        return llm, tokenizer

    monkeypatch.setattr(
        "llamea.llm.load",
        fake_load
    )

    return {
        "llm": llm,
        "tokenizer": tokenizer,
        "calls": calls,
    }

class AlwaysFailLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("LLM failure")

class FakeTokenier:
    def apply_chat_template(self, session, add_generation_prompt:bool):
        return session


def test_query_fails_only_after_max_tries(fake_mlx_load, monkeypatch):

    mock_tokenizer = FakeTokenier()
    failing_llm = AlwaysFailLLM()
    # Replace the llm returned by load()
    fake_mlx_load["llm"] = failing_llm

    llm = MLX_LM_LLM(
        model="dummy-model",
        max_tokens=10,
    )

    # monkeypatch the internal llm and dependencies
    monkeypatch.setattr('llamea.llm.generate', failing_llm.generate)
    llm.tokenizer = mock_tokenizer

    result = llm.query([{"client": "hello"}], max_tries=3, add_generation_prompt=True)

    assert result == ""
    assert failing_llm.calls == 3

def test_deepcopy_shares_llm_and_tokenizer(fake_mlx_load):
    llm = MLX_LM_LLM(
        model="dummy-model",
        config={"foo": "bar"},
        max_tokens=123,
    )

    llm_copy = copy.deepcopy(llm)

    # Shared heavy objects
    assert llm.llm is llm_copy.llm
    assert llm.tokenizer is llm_copy.tokenizer

    # But normal attributes are copied
    assert llm is not llm_copy
    assert llm.config == llm_copy.config
    assert llm.max_tokens == llm_copy.max_tokens

def test_getstate_setstate_restores_llm_and_tokenizer(fake_mlx_load):
    llm = MLX_LM_LLM(
        model="dummy-model",
        config={"alpha": 0.5},
        max_tokens=42,
    )

    # Serialize + deserialize
    blob = pickle.dumps(llm)
    restored = pickle.loads(blob)

    # load() must be called again
    assert len(fake_mlx_load["calls"]) == 2

    # Config and parameters restored
    assert restored.model == llm.model
    assert restored.config == llm.config
    assert restored.max_tokens == llm.max_tokens

    # Heavy objects exist
    assert hasattr(restored, "llm")
    assert hasattr(restored, "tokenizer")

    assert restored.llm is fake_mlx_load["llm"]
    assert restored.tokenizer is fake_mlx_load["tokenizer"]

class AlwaysFailLMStudio:
    def __init__(self):
        self.calls = 0

    def respond(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("LMStudio failure")


class AlwaysSucceedLMStudio:
    def __init__(self, response="OK"):
        self.calls = 0
        self.response = response

    def respond(self, *args, **kwargs):
        self.calls += 1
        return self.response

import types, sys
@pytest.fixture
def fake_lms_llm(monkeypatch):
    calls = []

    lms = types.ModuleType("llamea.llm.lms")

    def factory(model):
        client = AlwaysFailLMStudio()
        calls.append((model, client))
        return client

    lms.llm = factory

    monkeypatch.setitem(sys.modules, "llamea.llm.lms", lms)

    import llamea.llm
    monkeypatch.setattr(llamea.llm, "lms", lms, raising=False)

    return {"calls": calls}

def test_lms_query_fails_only_after_max_tries(fake_lms_llm):
    llm = LMStudio_LLM(
        model="dummy-model",
        config=None,
    )

    # Replace llm with failing instance explicitly
    failing = AlwaysFailLMStudio()
    llm.llm = failing

    result = llm.query(
        [{"role": "user", "content": "hello"}],
        max_tries=3,
    )

    assert result == ""
    assert failing.calls == 3

def test_lms_query_success_and_think_stripped(fake_lms_llm):
    llm = LMStudio_LLM(model="dummy", config=None)

    llm.llm = AlwaysSucceedLMStudio(
        response="<think>This client dare only greet me??? Fine, I'll be kind for now, *hmph*.</think>Hello!"
    )

    result = llm.query(
        [{"role": "user", "content": "hi"}],
        max_tries=2,
    )

    assert result == "Hello!"

def test_lms_deepcopy_shares_llm_instance(fake_lms_llm):
    llm = LMStudio_LLM(
        model="dummy-model",
        config={"temperature": 0.2},
    )

    llm_copy = copy.deepcopy(llm)

    # Shared heavy object
    assert llm.llm is llm_copy.llm

    # Independent wrapper
    assert llm is not llm_copy
    assert llm.model == llm_copy.model
    assert llm.config == llm_copy.config

def test_lms_getstate_setstate_reloads_llm(fake_lms_llm):
    llm = LMStudio_LLM(
        model="dummy-model",
        config={"top_p": 0.9},
    )

    # One load from __init__
    assert len(fake_lms_llm["calls"]) == 1

    blob = pickle.dumps(llm)
    restored = pickle.loads(blob)

    # Second load from __setstate__
    assert len(fake_lms_llm["calls"]) == 2

    assert restored.model == llm.model
    assert restored.config == llm.config
    assert hasattr(restored, "llm")

def test_default_models_are_current():
    """Defaults must not name models the providers have shut down (#165)."""
    from llamea import llm as llm_module
    from llamea.llm import DeepSeek_LLM, Gemini_LLM, Ollama_LLM, OpenAI_LLM

    assert OpenAI_LLM(api_key="sk-test").model == llm_module.DEFAULT_OPENAI_MODEL
    assert Gemini_LLM(api_key="key").model == llm_module.DEFAULT_GEMINI_MODEL
    assert Ollama_LLM().model == llm_module.DEFAULT_OLLAMA_MODEL
    assert DeepSeek_LLM(api_key="key").model == llm_module.DEFAULT_DEEPSEEK_MODEL
    retired = {"gpt-4-turbo", "gemini-2.0-flash", "deepseek-chat", "llama3.2"}
    defaults = {
        llm_module.DEFAULT_OPENAI_MODEL,
        llm_module.DEFAULT_GEMINI_MODEL,
        llm_module.DEFAULT_OLLAMA_MODEL,
        llm_module.DEFAULT_DEEPSEEK_MODEL,
    }
    assert not defaults & retired


def test_openai_omits_temperature_by_default():
    from llamea.llm import DeepSeek_LLM, OpenAI_LLM

    assert OpenAI_LLM(api_key="sk-test").temperature is None
    assert DeepSeek_LLM(api_key="key").temperature == 0.8
