import requests
import json
from dotenv import load_dotenv
import os


load_dotenv()


class OpenAIChat:
    def __init__(
            self,
            openai_api_key: str | None = None,
            openai_model: str = 'gpt-3.5-turbo',
            embedding_model: str = 'text-embedding-3-small',
            response_keys: tuple = ('text', 'reference_url', 'image_url')
    ):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_model = openai_model
        self.assistant = None
        self.embedding_model = embedding_model
        self.url = 'https://api.openai.com'
        self.messages = [
            {
                'role': 'system',
                'content': f'You are a helpful assistant designed to output JSON. Use these keys only if related: {", ".join(response_keys)}'
            }
        ]

    def _post_request(self, endpoint: str, headers: dict, data: dict) -> dict:
        try:
            response = requests.post(f'{self.url}/{endpoint}', headers=headers, json=data, timeout=20)
        except requests.exceptions.RequestException as err:
            return {'status': f'Error: {err}'}
        else:
            result = {'status': response.status_code}
            if result['status'] == 200:
                result['json'] = response.json()
            return result

    def _get_request(self, endpoint: str, headers: dict) -> dict:
        try:
            response = requests.get(f'{self.url}/{endpoint}', headers=headers, timeout=20)
        except requests.exceptions.RequestException as err:
            return {'status': f'Error: {err}'}
        else:
            result = {'status': response.status_code}
            if result['status'] == 200:
                result['json'] = response.json()
            return result

    def send_message(self, message: str):
        self.add_message('user', message)
        return self.get_response()

    def add_message(self, role: str, content: str):
        self.messages.append({'role': role, 'content': content})

    def get_response(self) -> dict:
        end_point = 'v1/chat/completions'
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        data = {
            'model': self.openai_model,
            'response_format': {'type': 'json_object'},
            'messages': self.messages
        }
        response = self._post_request(end_point, headers, data)

        if not (result := response.get('json')):
            return {'text': response.get('status')}

        if 'choices' in result and len(result['choices']) > 0:
            last_message = result['choices'][-1]['message']['content']
            self.add_message('assistant', last_message)
            return json.loads(last_message)
        else:
            return {'text': 'No response from OpenAI API'}



    def create_embeddings(self, text) -> list:
        endpoint = 'v1/embeddings'
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        data = {
            'input': text,
            'model': 'text-embedding-3-small',
            'encoding_format': 'float'
        }

        response = self._post_request(endpoint, headers, data)

        if not (res := response.get('json')):
            return []

        return res['data'][0]['embedding']


def main():
    openai = OpenAIChat(openai_model='gpt-4o-mini')
    # print(openai.send_message('Hello, how are you?'))
    print(openai.create_embeddings('hello'))


if __name__ == '__main__':
    main()
