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
     - ``OpenAI_LLM(api_key, "gpt-6.2-sol")``, ``OpenAI_LLM(key, "model", base_url="http://localhost:8000/v1")``
   * - :class:`~llamea.llm.Gemini_LLM`
     - Google Gemini
     - ``Gemini_LLM(api_key, "gemini-3.8-flash")``
   * - :class:`~llamea.llm.Anthropic_LLM`
     - Anthropic Claude
     - ``Anthropic_LLM(api_key, "claude-sonnet-5-5")``
   * - :class:`~llamea.llm.DeepSeek_LLM`
     - DeepSeek
     - ``DeepSeek_LLM(api_key, "deepseek-flash")``
   * - :class:`~llamea.llm.Ollama_LLM`
     - Local models through Ollama
     - ``Ollama_LLM("qwen2.5-coder:7b")``
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

Default models
--------------

Each wrapper has a default model, used when you pass none. Model names change
often, so name the model explicitly in experiments you want to reproduce.

.. list-table::
   :header-rows: 1

   * - Class
     - Default (since 1.6.0)
   * - ``OpenAI_LLM``
     - ``gpt-6.2-sol`` (``gpt-6.2-astra`` is the flagship, ``gpt-6-luna`` the cheapest)
   * - ``Gemini_LLM``
     - ``gemini-3.8-flash``
   * - ``Anthropic_LLM``
     - ``claude-sonnet-5-5``
   * - ``DeepSeek_LLM``
     - ``deepseek-flash`` (``deepseek-v4-pro`` is the larger model)
   * - ``Ollama_LLM``
     - ``qwen2.5-coder:7b`` (``qwen3-coder:30b`` on a larger GPU)

Temperature
-----------

``OpenAI_LLM`` leaves ``temperature`` out of the request by default, as OpenAI
recommends for its current models, some of which reject other values. Pass
``temperature=0.8`` (for example) for a model that accepts it.

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
