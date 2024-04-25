import os
from openai import AsyncOpenAI
import tiktoken

from .model import ModelProvider
from .openai import OpenAI


class Stepfun(OpenAI):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=1999, temperature=0.01)

    def __init__(self, api_key: str = "", model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        self.model_name = api_key
        self.model_kwargs = model_kwargs
        self.api_key = os.getenv('STEPFUN_API_KEY')
        self.model = AsyncOpenAI(api_key=self.api_key, base_url="https://api.stepfun.com/v1")
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')

    async def evaluate_model(self, prompt: str) -> str:
        response = await self.model.chat.completions.create(
            model='step-1-200k',
            messages=prompt,
            **self.model_kwargs
        )
        return response.choices[0].message.content
