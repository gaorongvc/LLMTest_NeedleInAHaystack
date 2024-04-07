import os
from operator import itemgetter
from typing import Optional

from openai import AsyncOpenAI
from langchain.prompts import PromptTemplate
import tiktoken

from .model import ModelProvider
from .openai import OpenAI
from zhipuai import ZhipuAI


class Zhipu(OpenAI):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=300, temperature=0.01)

    def __init__(self, api_key: str = "", model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        self.model_name = api_key
        self.model_kwargs = model_kwargs
        self.api_key = '7969f621c1fdaad9a35c354b9ab0db45.wdmQvzH0QV4IPEE6'
        self.model = ZhipuAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')

    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        print(prompt)
        print(self.model_kwargs)
        response = self.model.chat.completions.create(
            model='glm-4',
            messages=prompt,
            **self.model_kwargs
        )
        return response.choices[0].message.content
