import requests
import json
from dotenv import load_dotenv
import os
import chromadb
import tiktoken

import pdfplumber


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

class OpenAIChat:
    def __init__(
            self,
            openai_api_key: str | None = None,
            openai_model: str = 'gpt-3.5-turbo',
            embedding_model: str = 'text-embedding-3-small',
            response_keys: tuple = ('text', 'reference_url', 'image_url'),
            max_tokens: int = 512
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
        self.max_tokens = max_tokens
        self.chroma_client_temporary = chromadb.Client()
        self.chroma_client_persistent = chromadb.PersistentClient()
        self.collection_temporary = self.chroma_client_temporary.get_or_create_collection(name="embeddings_collection")
        self.collection_persistent = self.chroma_client_persistent.get_or_create_collection(name="embeddings_collection")

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

    def send_message(self, message: str):
        embedding = self.create_embeddings(message)
        if not embedding:
            return {'text': 'Failed to generate embedding for the message.'}

        self.add_message('user', message)

        results = self.collection_persistent.query(
            query_embeddings=[embedding],
            n_results=3
        )

        similar_texts = []
        for result in results['documents']:
            if result:
                similar_texts.extend(result)

        if similar_texts:
            context = " ".join([text for text in similar_texts if text])
            self.add_message('system', f"Previously, you said: {context}")

        response = self.get_response()

        return response

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

    def chunk_text(self, text):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        # Split tokens into chunks
        chunks = [tokens[i:i + self.max_tokens] for i in range(0, len(tokens), self.max_tokens)]
        chunked_texts = [encoding.decode(chunk) for chunk in chunks]
        return chunked_texts

    def create_embeddings_for_long_text(self, text):
        chunked_texts = self.chunk_text(text)
        embeddings = []
        for chunk in chunked_texts:
            embedding = self.create_embeddings(chunk)
            if embedding:
                embeddings.append(embedding)

        # Average the embeddings if multiple chunks
        if embeddings:
            averaged_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
            return averaged_embedding
        else:
            return []

    def create_embeddings(self, text) -> list:
        endpoint = 'v1/embeddings'
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        data = {
            'input': text,
            'model': self.embedding_model,
            'encoding_format': 'float'
        }

        response = self._post_request(endpoint, headers, data)

        if not (res := response.get('json')) or 'data' not in res or not res['data']:
            return []

        embedding = res['data'][0]['embedding']
        return embedding

    def store_embedding_temporary(self, text: str, text_id: str, metadata: dict = None):
        embedding = self.create_embeddings_for_long_text(text)
        if not embedding:
            print(f"Failed to generate embedding for text_id {text_id}")
            return

        self.collection_temporary.add(
            embeddings=[embedding],
            ids=[text_id],
            metadatas=[metadata] if metadata else [{}],
            documents=[text]
        )
        print(f"Embedding for text_id {text_id} stored successfully.")

        return embedding

    def store_embedding_persistent(self, text: str, text_id: str, metadata: dict = None):
        embedding = self.create_embeddings_for_long_text(text)
        if not embedding:
            print(f"Failed to generate embedding for text_id {text_id}")
            return

        self.collection_persistent.add(
            embeddings=[embedding],
            ids=[text_id],
            metadatas=[metadata] if metadata else [{}],
            documents=[text]
        )
        print(f"Embedding for text_id {text_id} stored successfully.")

        return embedding



def pdf_to_text(pdf_document):
    doc_id = 'validated-document'
    text = ''

    with pdfplumber.open(pdf_document) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return {'id': doc_id, 'text': text}


def main():
    openai = OpenAIChat(openai_model='gpt-4o-mini')

    validated_pdf = pdf_to_text('merged_files.pdf')
    openai.store_embedding_persistent(validated_pdf['text'], str(validated_pdf['id']), {'source': 'pre_validated_pdf'})

    result = openai.collection_persistent.get(ids=['validated-document'], include=['embeddings', 'documents'])
    print(result)

    # response = openai.send_message("What is your context?")
    # print(response['text'])


if __name__ == '__main__':
    main()
