import os
import dashscope
from typing import Optional

from .model import ModelProvider

class QianWenProvider(ModelProvider):
    """
    A command-line interface for interacting with Qianwen's API for question answering tasks.
    """

    def __init__(self, model_name: str = "qwen_max", api_key: Optional[str] = None):
        """
        Initialize the Qianwen provider with the specified model and API key.

        Args:
            model_name (str): Name of the Qianwen model to use. Defaults to 'qwen_max'.
            api_key (Optional[str]): The API key for authentication. If not provided, it will use the value from the environment variable `DASHSCOPE_API_KEY`.
        """
        if api_key:
            dashscope.api_key = api_key
        else:
            dashscope.api_key = os.getenv("sk-2205d049f5714439bb1d849c32cb814a")
            if not dashscope.api_key:
                raise ValueError("DASHSCOPE_API_KEY must be set in the environment or provided explicitly.")

        self.model_name = model_name
        if model_name == "qwen_max":
            self.model = dashscope.Generation.Models.qwen_max
        else:
            raise ValueError(f"model_name {model_name} is not supported")

    def generate_prompt(self, context: str, retrieval_question: str) -> list[dict[str, str]]:
        """
        Generate a prompt for the model based on the provided context and question.

        Args:
            context (str): Background context for the question.
            retrieval_question (str): The question to answer.

        Returns:
            A list of dictionaries representing the conversation for the model prompt.
        """
        return [
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided context."},
            {"role": "user", "content": context},
            {"role": "user", "content": f"{retrieval_question} Don't provide information outside the given context or repeat your findings."}
        ]

    def run(self, context: str, retrieval_question: str) -> str:
        """
        Run the model on the given context and question.

        Args:
            context (str): Background context for the question.
            retrieval_question (str): The question to answer.

        Returns:
            The model's response as a string.
        """
        prompt = self.generate_prompt(context, retrieval_question)
        response = dashscope.Generation.call(
            self.model,
            messages=prompt,
            result_format="message",
        )
        return response.output.choices[0].message.content

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Qianwen model for question answering.")
    parser.add_argument("--model_name", type=str, default="qwen_max", help="Name of the Qianwen model to use.")
    parser.add_argument("--api_key", type=str, help="API key for authentication. If not provided, it will use the value from the environment variable `DASHSCOPE_API_KEY`.")
    parser.add_argument("--context", type=str, required=True, help="Background context for the question.")
    parser.add_argument("--retrieval_question", type=str, required=True, help="The question to answer.")

    args = parser.parse_args()

    provider = QianWenProvider(model_name=args.model_name, api_key=args.api_key)
    answer = provider.run(args.context, args.retrieval_question)
    print(answer)