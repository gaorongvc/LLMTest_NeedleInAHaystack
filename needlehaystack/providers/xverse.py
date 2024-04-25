import os
import requests
import tiktoken

from .model import ModelProvider
from .openai import OpenAI


class Xverse(OpenAI):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=1999, temperature=0.01)

    def __init__(self, api_key: str = "", model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        self.model_name = api_key
        self.model_kwargs = model_kwargs
        self.api_key = os.getenv('XVERSE_API_KEY')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')

    async def evaluate_model(self, prompt: str) -> str:
        json_data = {
            'messages': prompt,
            'plugin': 'MODEL',
            'model': 'XVERSE-13B-LONGCONTEXT',
        }
        print(prompt)
        json_data.update(self.model_kwargs)
        response = requests.post('https://api.xverse.cn/v1/chat/completions', headers=self.headers, json=json_data)
        print(response.json())
        return response.json()['choices'][0]['message']['content']
