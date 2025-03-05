"""
LLM modules to connect to different LLM providers.
"""
from abc import ABC, abstractmethod
import google.generativeai as genai
import openai
import ollama

class LLM(ABC):
    def __init__(self, api_key, model='', base_url=''):
        """
        Initializes the LLM manager with an API key, model name and base_url.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation.
            base_url (str, optional): The url to call the API from.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

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

class OpenAI_LLM(LLM):
    """
    A manager class for handling requests to OpenAI's GPT models.
    """

    def __init__(self, api_key, model="gpt-4-turbo"):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, and others from OpeNAI models library.
        """
        super().__init__(api_key, model, None)
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

    def __init__(self, api_key, model="gemini-2.0-flash"):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gemini-2.0-flash".
                Options are: "gemini-1.5-flash","gemini-2.0-flash", and others from Googles models library.
        """
        super().__init__(api_key, model, None)
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
    def __init__(self, model="llama3.2"):
        """
        Initializes the Ollama LLM manager with a model name. See https://ollama.com/search for models.

        Args:
            model (str, optional): model abbreviation. Defaults to "llama3.2".
                See for options: https://ollama.com/search.
        """
        super().__init__('', model, None)


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