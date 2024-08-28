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
            response_keys: tuple = ('text', 'reference_url', 'image_url')
    ):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_model = openai_model
        self.url = 'https://api.openai.com/v1/chat/completions'
        self.messages = [
            {
                'role': 'system',
                'content': f'You are a helpful assistant designed to output JSON. Use these keys only if related: {", ".join(response_keys)}'
            }
        ]

    def post_request(self, url: str, headers: dict, data: dict) -> dict:
        try:
            response = requests.post(url, headers=headers, json=data, timeout=20)
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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        data = {
            'model': self.openai_model,
            'response_format': {'type': 'json_object'},
            'messages': self.messages
        }
        response = self.post_request(self.url, headers, data)

        if not (result := response.get('json')):
            return {'text': response.get('status')}

        if 'choices' in result and len(result['choices']) > 0:
            last_message = result['choices'][-1]['message']['content']
            self.add_message('assistant', last_message)
            return json.loads(last_message)
        else:
            return {'text': 'No response from OpenAI API'}



def main():
    openai = OpenAIChat()
    print(openai.send_message('Hello, how are you?'))




if __name__ == '__main__':
    main()
