"""LLM manager to connect to different types of models.
"""
import google.generativeai as genai
import ollama
import openai


class LLMmanager:
    """
    A manager class for handling requests to multiple LLM providers, including
    OpenAI's GPT, Google Gemini, and Ollama-based models.
    """

    def __init__(self, api_key, model="gpt-4-turbo"):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): The API key for authenticating with the chosen LLM provider.
            model (str, optional): The model name or abbreviation, e.g. "gpt-4-turbo",
                "gpt-3.5-turbo", "gpt-4o", "gemini", "ollama", etc. Defaults to "gpt-4-turbo".
        """
        self.api_key = api_key
        self.model = model
        if "gpt" in self.model:
            self.client = openai.OpenAI(api_key=api_key)
        if "gemini" in self.model:
            genai.configure(api_key=api_key)
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            self.client = genai.GenerativeModel(
                model_name=self.model,  # "gemini-1.5-flash",
                generation_config=generation_config,
                # safety_settings = Adjust safety settings
                # See https://ai.google.dev/gemini-api/docs/safety-settings
                system_instruction="You are a computer scientist and excellent Python programmer.",
            )

    def chat(self, session_messages):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        if "gpt" in self.model:
            response = self.client.chat.completions.create(
                model=self.model, messages=session_messages, temperature=0.8
            )
            return response.choices[0].message.content
        elif "gemini" in self.model:
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
        else:
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
