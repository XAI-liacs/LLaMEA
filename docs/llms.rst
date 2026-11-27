LLM back-ends
=============

LLaMEA talks to language models through small wrapper classes in
:mod:`llamea.llm`. All share one interface: ``query(messages) -> str``.

.. list-table::
   :header-rows: 1

   * - Class
     - Provider
     - Example
   * - :class:`~llamea.llm.OpenAI_LLM`
     - OpenAI, or any OpenAI-compatible endpoint (vLLM, OpenRouter, ...)
     - ``OpenAI_LLM(api_key, "gpt-4o")``, ``OpenAI_LLM(key, "model", base_url="http://localhost:8000/v1")``
   * - :class:`~llamea.llm.Gemini_LLM`
     - Google Gemini
     - ``Gemini_LLM(api_key, "gemini-2.0-flash")``
   * - :class:`~llamea.llm.Anthropic_LLM`
     - Anthropic Claude
     - ``Anthropic_LLM(api_key, "claude-sonnet-5-5")``
   * - :class:`~llamea.llm.DeepSeek_LLM`
     - DeepSeek
     - ``DeepSeek_LLM(api_key, "deepseek-chat")``
   * - :class:`~llamea.llm.Ollama_LLM`
     - Local models through Ollama
     - ``Ollama_LLM("qwen2.5-coder:14b")``
   * - :class:`~llamea.llm.LMStudio_LLM`
     - Local models through LM Studio (Apple silicon extra)
     - ``LMStudio_LLM("model-name")``
   * - :class:`~llamea.llm.MLX_LM_LLM`
     - Local models through MLX (Apple silicon extra)
     - ``MLX_LM_LLM("mlx-community/...")``
   * - :class:`~llamea.llm.Multi_LLM`
     - Picks one of several LLMs at random per query
     - ``Multi_LLM([llm_a, llm_b])``
   * - :class:`~llamea.llm.Dummy_LLM`
     - No model; fixed response, for tests
     - ``Dummy_LLM()``

Reasoning models
----------------

Some reasoning models reject a ``temperature`` other than the default; create
``OpenAI_LLM(..., temperature=None)`` to leave it out of the request.

Errors and retries
------------------

Rate limits, timeouts, connection and server errors are retried with a
growing delay (honouring ``Retry-After``). Permanent errors such as a wrong
API key, an unknown model or an invalid parameter fail immediately.

How responses are parsed
------------------------

LLaMEA takes the first Python (or untagged) fenced code block of the reply as
the code, the first ``class`` (else ``def``) as the solution's name, and the
``# Description:`` line as its description. In diff mode, every fenced block
with SEARCH/REPLACE markers is applied. You can pass your own ``code_pattern``,
``name_pattern``, ``desc_pattern`` or ``cs_pattern`` (regular expressions) to
any LLM class.

Your own LLM
------------

Subclass :class:`~llamea.llm.LLM` and implement ``query``:

.. code-block:: python

   from llamea import LLM

   class MyLLM(LLM):
       def __init__(self, model="my-model", **kwargs):
           super().__init__(api_key="", model=model, **kwargs)

       def query(self, session_messages):
           # session_messages: [{"role": "user", "content": "..."}, ...]
           return my_client.chat(self.model, session_messages)
