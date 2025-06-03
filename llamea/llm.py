"""
LLM modules to connect to different LLM providers. Also extracts code, name and description.
"""
import re
from abc import ABC, abstractmethod

import google.generativeai as genai
import ollama
import openai
from ConfigSpace import ConfigurationSpace

from .solution import Solution
from .utils import NoCodeException


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
            code_pattern if code_pattern != None else r"```(?:python)?\n(.*?)\n```"
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
            session_messages (list of dict): A list of message dictionaries with keys
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

    def sample_solution(self, session_messages: list, parent_ids=[], HPO=False):
        """
        Interacts with a language model to generate or mutate solutions based on the provided session messages.

        Args:
            session_messages (list): A list of dictionaries with keys 'role' and 'content' to simulate a conversation with the language model.
            parent_ids (list, optional): The id of the parent the next sample will be generated from (if any).
            HPO (boolean, optional): If HPO is enabled, a configuration space will also be extracted (if possible).

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        if self.log:
            self.logger.log_conversation(
                "client", "\n".join([d["content"] for d in session_messages])
            )

        message = self.query(session_messages)

        if self.log:
            self.logger.log_conversation(self.model, message)

        code = self.extract_algorithm_code(message)
        name = re.findall(
            "class\\s*(\\w*)(?:\\(\\w*\\))?\\:",
            code,
            re.IGNORECASE,
        )[0]
        desc = self.extract_algorithm_description(message)
        cs = None
        if HPO:
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
        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                c = ConfigurationSpace(eval(m.group(1)))
            except Exception as e:
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
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
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

    def __init__(self, api_key, model="gpt-4-turbo", **kwargs):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, and others from OpeNAI models library.
        """
        super().__init__(api_key, model, None, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)

    def query(self, session_messages):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """

        response = self.client.chat.completions.create(
            model=self.model, messages=session_messages, temperature=0.8
        )
        return response.choices[0].message.content


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
        super().__init__(api_key, model, None, **kwargs)
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.client = genai.GenerativeModel(
            model_name=self.model,  # "gemini-1.5-flash","gemini-2.0-flash",
            generation_config=generation_config,
            system_instruction="You are a computer scientist and excellent Python programmer.",
        )

    def query(self, session_messages):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """

        history = []
        last = session_messages.pop()
        for msg in session_messages:
            history.append(
                {
                    "role": msg["role"],
                    "parts": [
                        msg["content"],
                    ],
                }
            )
        chat_session = self.client.start_chat(history=history)
        response = chat_session.send_message(last["content"])
        return response.text


class Ollama_LLM(LLM):
    def __init__(self, model="llama3.2", **kwargs):
        """
        Initializes the Ollama LLM manager with a model name. See https://ollama.com/search for models.

        Args:
            model (str, optional): model abbreviation. Defaults to "llama3.2".
                See for options: https://ollama.com/search.
        """
        super().__init__("", model, None, **kwargs)

    def query(self, session_messages):
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
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": big_message,
                }
            ],
        )
        return response["message"]["content"]
