#!/usr/bin/env python
"""
API client utilities for GPT, Gemini, and other API-based models.
"""
import base64
import json
import logging
import time
from typing import Optional
from openai import OpenAI


def parse_gpt_response(response):
    """Parse GPT response to extract JSON data."""
    try:
        if "```json" in response.lower():
            json_str = response.lower().split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1]
        else:
            json_str = response

        operations = json.loads(json_str)

        if isinstance(operations, (dict, list)):
            return operations
        else:
            raise ValueError("The operations format in GPT response is not a list or dict.")

    except Exception as e:
        logging.error("Unable to parse GPT response into JSON.")
        logging.debug(f"Original response content: {response}")
        raise e


class Client:
    def __init__(
        self,
        base_url: str,
        max_retries: int = 50,
        max_tokens: int = 4096,
        model_name: str = "",
    ):
        self.base_url = base_url
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.model_name = model_name

    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def handle_retries(self, func, *args, **kwargs):
        """Helper to handle retries for API calls."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = func(*args, **kwargs)
                return response
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(2)
        raise Exception("Max retries reached. Request failed.")


class GPTClient(Client):
    def __init__(
        self,
        api_key: str,
        base_url: str = "#TODO",  # Your API base URL
        model_name: str = "gpt-4o",
        max_retries: int = 50,
        max_tokens: int = 4096,
    ):
        super().__init__(base_url, max_retries, max_tokens, model_name)
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(
        self, img_path: Optional[str] = None, query: str = "", **kwargs
    ) -> Optional[str]:
        """Handle both image and text-based queries."""
        if img_path:
            base64_image = self.encode_image(img_path)
            if not base64_image:
                return None
            return self.handle_retries(self.analyze_image, base64_image, query)
        else:
            return self.handle_retries(self.analyze_text, query)

    def analyze_image(self, base64_image: str, query: str) -> str:
        """Send the image and query to OpenAI for analysis."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": query}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        ],
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error during '{self.model_name}' request: {e}")

    def analyze_text(self, query: str) -> str:
        """Send only the text query to OpenAI for analysis."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": query}],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error during '{self.model_name}' request: {e}")


class GPT4oMiniClient(GPTClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "#TODO",
        model_name: str = "gpt-4o-mini",
        max_retries: int = 50,
        max_tokens: int = 4096,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            max_tokens=max_tokens,
            model_name=model_name,
        )


class GeminiClient(GPTClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "#TODO",
        model_name: str = "gemini-2.5-pro-preview-05-06",
        max_retries: int = 50,
        max_tokens: int = 4096,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            max_tokens=max_tokens,
            model_name=model_name,
        )


class Qwen25VL72BClient(GPTClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        max_retries: int = 50,
        max_tokens: int = 4096,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            max_tokens=max_tokens,
            model_name=model_name,
        )


ClientMap = {
    "gemini": GeminiClient,
    "gpt4o": GPTClient,
    "gpt4o-mini": GPT4oMiniClient,
    "qwenvl2.5-72B": Qwen25VL72BClient,
}
