import tiktoken

from .openai import OpenAI
from dify_client import ChatClient


class DifyX(OpenAI):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens=300, temperature=0)

    def __init__(self, api_key: str = "", model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        self.model_name = api_key
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        self.model = ChatClient(self.api_key)
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')

    async def evaluate_model(self, prompt: str) -> str:
        response = self.model.create_chat_message(inputs={}, query=prompt, user="needleinhaystack",
                                                  response_mode="blocking")
        response.raise_for_status()
        content = response.json()['answer']
        return content

    async def evaluate_model_2(self, prompt: str) -> str:
        response = self.model.create_chat_message(inputs={}, query=prompt, user="needleinhaystack",
                                                  response_mode="streaming")
        response.raise_for_status()
        content = ""
        for line in chat_response.iter_lines(decode_unicode=True):
            line = line.split('data:', 1)[-1]
            if line.strip():
                line = json.loads(line.strip())
                content += line.get('answer')
        return content
