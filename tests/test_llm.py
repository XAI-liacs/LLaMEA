import copy
import datetime as _dt
import pickle
import random
from unittest.mock import MagicMock, patch

import httpx
import pytest

import llamea.llm as llm_mod  # the module that defines query
from llamea import (
    LLM,
    Dummy_LLM,
    Gemini_LLM,
    Multi_LLM,
    Ollama_LLM,
    OpenAI_LLM,
    Multi_LLM,
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


def test_llm_sample_solution_diff_patch():
    class DummyLLM(LLM):
        def query(self, session: list):
            return (
                "# Description: Modified\n"
                "```diff\n"
                "--- original.py\n"
                "+++ updated.py\n"
                "@@ -1,2 +1,3 @@\n"
                " class MyAlgo:\n"
                "-    pass\n"
                "+    def run(self):\n"
                "+        return 42\n"
                "```"
            )

    base = "class MyAlgo:\n    pass\n"
    llm = DummyLLM(api_key="x", model="y")
    sol = llm.sample_solution(
        [{"role": "client", "content": "test"}],
        base_code=base,
        diff_mode=True,
    )
    assert "return 42" in sol.code
    assert sol.name == "MyAlgo"


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
    fake_client.start_chat.side_effect = [chat_fail, chat_ok]
    llm.client = fake_client

    reply = llm.query([{"role": "user", "content": "hello"}], max_retries=3)

    assert reply == "OK-DONE"
    assert fake_client.start_chat.call_count == 2  # 1 failure + 1 success
    slept.assert_called_once_with(3)  # 2 s + 1 s safety buffer


def test_gemini_llm_gives_up_after_max_retries(monkeypatch):
    """query should bubble the error once max_retries is exceeded."""
    llm = Gemini_LLM(api_key="fake", model="gemini-test")

    slept = MagicMock()
    monkeypatch.setattr(llm_mod.time, "sleep", slept)

    chat_fail = MagicMock()
    chat_fail.send_message.side_effect = _resource_exhausted(1)

    fake_client = MagicMock()
    fake_client.start_chat.return_value = chat_fail
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
