import os
import tiktoken

from .openai import OpenAI
from zhipuai import ZhipuAI


class Zhipu(OpenAI):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=300, temperature=0.01)

    def __init__(self, api_key: str = "", model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        self.model_name = api_key
        self.model_kwargs = model_kwargs
        self.api_key = os.getenv('ZHIPU_API_KEY')
        self.model = ZhipuAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')

    async def evaluate_model(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model='glm-4',
            messages=prompt,
            **self.model_kwargs
        )
        return response.choices[0].message.content
