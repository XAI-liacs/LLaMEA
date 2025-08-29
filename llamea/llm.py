"""
LLM modules to connect to different LLM providers. Also extracts code, name and description.
"""

import copy
import logging
import pickle
import json
import random
import re
import time
from abc import ABC, abstractmethod

try:
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    genai = None

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ollama = None

try:
    import openai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None

try:
    from ConfigSpace import ConfigurationSpace
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ConfigurationSpace = None

from .solution import Solution
from .loggers import ExperimentLogger
from .utils import NoCodeException, apply_unified_diff


class LLM(ABC):
    def __init__(
        self,
        api_key,
        model="",
        base_url="",
        code_pattern=None,
        name_pattern=None,
        desc_pattern=None,
        cs_pattern=None,
        logger=None,
    ):
        """
        Initializes the LLM manager with an API key, model name and base_url.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation.
            base_url (str, optional): The url to call the API from.
            code_pattern (str, optional): The regex pattern to extract code from the response.
            name_pattern (str, optional): The regex pattern to extract the class name from the response.
            desc_pattern (str, optional): The regex pattern to extract the description from the response.
            cs_pattern (str, optional): The regex pattern to extract the configuration space from the response.
            logger (Logger, optional): A logger object to log the conversation.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.log = self.logger != None
        self.code_pattern = (
            code_pattern
            if code_pattern is not None
            else r"```(?:python|diff)?\n(.*?)\n```"
        )
        self.name_pattern = (
            name_pattern
            if name_pattern != None
            else "class\\s*(\\w*)(?:\\(\\w*\\))?\\:"
        )
        self.desc_pattern = (
            desc_pattern if desc_pattern != None else r"#\s*Description\s*:\s*(.*)"
        )
        self.cs_pattern = (
            cs_pattern
            if cs_pattern != None
            else r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        )

    @abstractmethod
    def query(self, session: list):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        pass

    def set_logger(self, logger):
        """
        Sets the logger object to log the conversation.

        Args:
            logger (Logger): A logger object to log the conversation.
        """
        self.logger = logger
        self.log = True

    def sample_solution(
        self,
        session_messages: list,
        parent_ids: list | None = None,
        HPO: bool = False,
        base_code: str | None = None,
        diff_mode: bool = False,
    ):
        """Generate or mutate a solution using the language model.

        Args:
            session_messages: Conversation history for the LLM.
            parent_ids: Identifier(s) of parent solutions.
            HPO: If ``True``, attempt to extract a configuration space.
            base_code: Existing code to patch when ``diff_mode`` is ``True``.
            diff_mode: When ``True``, interpret the LLM response as a unified
                diff patch to apply to ``base_code`` rather than full source
                code.

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        if parent_ids is None:
            parent_ids = []

        if self.log:
            self.logger.log_conversation(
                "client", "\n".join([d["content"] for d in session_messages])
            )

        message = self.query(session_messages)

        if self.log:
            self.logger.log_conversation(self.model, message)

        code_block = self.extract_algorithm_code(message)
        if diff_mode:
            if base_code is None:
                base_code = ""
            code = apply_unified_diff(base_code, code_block)
        else:
            code = code_block

        name = re.findall(
            "class\\s*(\\w*)(?:\\(\\w*\\))?\\:",
            code,
            re.IGNORECASE,
        )[0]
        desc = self.extract_algorithm_description(message)
        cs = None
        if HPO and ConfigurationSpace is not None:
            cs = self.extract_configspace(message)
        new_individual = Solution(
            name=name,
            description=desc,
            configspace=cs,
            code=code,
            parent_ids=parent_ids,
        )

        return new_individual

    def extract_configspace(self, message):
        """
        Extracts the configuration space definition in json from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            ConfigSpace: Extracted configuration space object.
        """
        if ConfigurationSpace is None:  # pragma: no cover - optional dependency
            return None
        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                c = ConfigurationSpace(eval(m.group(1)))
            except Exception as e:  # pragma: no cover - best effort
                logging.info(e)
                pass
        return c

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        match = re.search(self.code_pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            raise NoCodeException

    def extract_algorithm_description(self, message):
        """
        Extracts algorithm description from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"#\s*Description\s*:\s*(.*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""


class OpenAI_LLM(LLM):
    """
    A manager class for handling requests to OpenAI's GPT models.
    """

    def __init__(self, api_key, model="gpt-4-turbo", temperature=0.8, **kwargs):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, and others from OpeNAI models library.
        """
        if openai is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "openai is required to use OpenAI_LLM. Install the 'openai' package."
            )
        super().__init__(api_key, model, None, **kwargs)
        self._client_kwargs = dict(api_key=api_key)
        self.client = openai.OpenAI(**self._client_kwargs)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        self.temperature = temperature

    def query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """

        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=session_messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except openai.RateLimitError as err:
                attempt += 1
                if attempt > max_retries:
                    raise

                retry_after = None
                if getattr(err, "response", None) is not None:
                    retry_after = err.response.headers.get("Retry-After")

                wait = int(retry_after) if retry_after else default_delay * attempt
                time.sleep(wait)

            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIError,
            ) as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)

    # ---------- pickling / deepcopy helpers ----------
    def __getstate__(self):
        """Return the picklable part of the instance."""
        state = self.__dict__.copy()
        state.pop("client", None)  # the client itself is NOT picklable
        return state  # everything else is fine

    def __setstate__(self, state):
        """Restore from a pickled state."""
        self.__dict__.update(state)  # put back the simple stuff
        self.client = openai.OpenAI(
            **self._client_kwargs
        )  # rebuild non-picklable handle

    def __deepcopy__(self, memo):
        """Explicit deepcopy that skips the client and recreates it."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        # finally restore client
        new.client = openai.OpenAI(**new._client_kwargs)
        return new


class Gemini_LLM(LLM):
    """
    A manager class for handling requests to Google's Gemini models.
    """

    def __init__(self, api_key, model="gemini-2.0-flash", **kwargs):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gemini-2.0-flash".
                Options are: "gemini-1.5-flash","gemini-2.0-flash", and others from Googles models library.
        """
        if genai is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "google-generativeai is required to use Gemini_LLM. Install the 'google-generativeai' package."
            )
        super().__init__(api_key, model, None, **kwargs)
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
            "response_mime_type": "text/plain",
        }

        self.client = genai.GenerativeModel(
            model_name=self.model,  # "gemini-1.5-flash","gemini-2.0-flash",
            generation_config=generation_config,
            system_instruction="You are a computer scientist and excellent Python programmer.",
        )

    def query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends the conversation history to Gemini, retrying on 429 ResourceExhausted exceptions.

        Args:
            session_messages (list[dict]): [{"role": str, "content": str}, …]
            max_retries (int): how many times to retry before giving up.
            default_delay (int): fallback sleep when the error has no retry_delay.

        Returns:
            str: model's reply.
        """
        history = [
            {"role": m["role"], "parts": [m["content"]]} for m in session_messages[:-1]
        ]
        last = session_messages[-1]["content"]

        attempt = 0
        while True:
            try:
                chat = self.client.start_chat(history=history)
                response = chat.send_message(last)
                return response.text

            except Exception as err:
                attempt += 1
                if attempt > max_retries:
                    raise  # bubble out after N tries

                # Prefer the structured retry_delay field if present
                delay = getattr(err, "retry_delay", None)
                if delay is not None:
                    wait = delay.seconds + 1  # add 1 second to avoid immediate retry
                else:
                    # Sometimes retry_delay only appears in the string—grab it
                    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(err))
                    wait = int(m.group(1)) if m else default_delay * attempt

                time.sleep(wait)


class Ollama_LLM(LLM):
    def __init__(self, model="llama3.2", **kwargs):
        """
        Initializes the Ollama LLM manager with a model name. See https://ollama.com/search for models.

        Args:
            model (str, optional): model abbreviation. Defaults to "llama3.2".
                See for options: https://ollama.com/search.
        """
        if ollama is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "ollama is required to use Ollama_LLM. Install the 'ollama' package."
            )
        super().__init__("", model, None, **kwargs)

    def query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        # first concatenate the session messages
        big_message = ""
        for msg in session_messages:
            big_message += msg["content"] + "\n"

        attempt = 0
        while True:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": big_message}],
                )
                return response["message"]["content"]

            except ollama.ResponseError as err:
                attempt += 1
                if attempt > max_retries or err.status_code not in (429, 500, 503):
                    raise
                time.sleep(default_delay * attempt)

            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)


class Multi_LLM(LLM):
    def __init__(self, llms: list[LLM]):
        """
        Combine multiple LLM instances and randomly choose one per call.

        Args:
            llms (list[LLM]): A list of LLM instances to combine.
        """
        if not llms:
            raise ValueError("llms must contain at least one LLM instance")
        model = "multi-llm"
        super().__init__("", model)
        self.llms = llms

    def _pick_llm(self) -> LLM:
        """
        Randomly selects one of the LLMs from the list.
        This method is used to alternate between LLMs during evolution.
        """
        return random.choice(self.llms)

    def set_logger(self, logger):
        self.logger = logger
        self.log = True
        for llm in self.llms:
            llm.set_logger(logger)

    def query(self, session_messages: list):
        llm = self._pick_llm()
        return llm.query(session_messages)

    def sample_solution(self, *args, **kwargs):
        llm = self._pick_llm()
        return llm.sample_solution(*args, **kwargs)


class DeepSeek_LLM(OpenAI_LLM):
    """A manager class for the DeepSeek chat models."""

    def __init__(self, api_key, model="deepseek-chat", temperature=0.8, **kwargs):
        """Initializes DeepSeek LLM with required base URL."""
        super().__init__(api_key, model=model, temperature=temperature, **kwargs)
        self.base_url = "https://api.deepseek.com"
        self._client_kwargs["base_url"] = self.base_url
        self.client = openai.OpenAI(**self._client_kwargs)


class Dummy_LLM(LLM):
    def __init__(self, model="DUMMY", **kwargs):
        """
        Initializes the DUMMY LLM manager with a model name. This is a placeholder
        and does not connect to any LLM provider. It is used for testing purposes only.

        Args:
            model (str, optional): model abbreviation. Defaults to "DUMMY".
                Has no effect, just a placeholder.
        """
        super().__init__("", model, None, **kwargs)

    def query(self, session_messages):
        """
        Sends a conversation history to DUMMY model and returns a random response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        # first concatenate the session messages
        big_message = ""
        for msg in session_messages:
            big_message += msg["content"] + "\n"
        response = """This is a dummy response from the DUMMY LLM. It does not connect to any LLM provider.
It is used for testing purposes only.
# Description: A simple random search algorithm that samples points uniformly in the search space and returns the best found solution.
# Code:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```
# Configuration Space:
```python
{
    'budget': {'type': 'int', 'lower': 1000, 'upper': 100000},
    'dim': {'type': 'int', 'lower': 1, 'upper': 100}
}
```
"""
        return response
