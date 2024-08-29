import streamlit as st
from openai_helper import OpenAIChat
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import uuid
import numpy as np



def pdf_to_text(pdf_document):
    doc_id = uuid.uuid4()
    text = ''

    with pdfplumber.open(pdf_document) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return {'id': doc_id, 'text': text}


def pdf_to_vectordb(pdf_document):
    pdf = pdf_to_text(pdf_document)
    text_id = pdf['id']
    text = pdf['text']
    metadata = {'source': 'user_uploaded_pdf'}
    st.session_state.uploaded_pdf_embedding = st.session_state.chat.store_embedding(text, str(text_id), metadata)


def load_pre_validated_pdf(path):
    validated_pdf = pdf_to_text(path)
    text_id = validated_pdf['id']
    text = validated_pdf['text']
    metadata = {'source': 'pre_validated_pdf'}
    st.session_state.validated_pdf_embedding = st.session_state.chat.store_embedding(text, str(text_id), metadata)


# def compare_uploaded_pdf_with_context(uploaded_pdf):
#     uploaded_pdf_embedding = st.session_state.uploaded_pdf_embedding
#     validated_pdf_embedding = st.session_state.validated_pdf_embedding
#
#     cosine_similarity = np.dot(validated_pdf_embedding, uploaded_pdf_embedding) / (np.linalg.norm(validated_pdf_embedding) * np.linalg.norm(uploaded_pdf_embedding))
#     print(f"Cosine Similarity: {cosine_similarity}")
#     return cosine_similarity


def compare_uploaded_pdf_with_context(uploaded_pdf):
    uploaded_pdf_embedding = st.session_state.uploaded_pdf_embedding
    validated_pdf_embedding = st.session_state.validated_pdf_embedding

    # Note that sklearn expects 2D arrays
    cosine_sim = cosine_similarity([validated_pdf_embedding], [uploaded_pdf_embedding])
    return cosine_sim[0][0] * 100


st.title('\U0001F3E1 Chat with OpenAI RAG AI solution using PDF')

if 'chat' not in st.session_state:
    st.session_state.chat = OpenAIChat(openai_model='gpt-4o-mini')
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I help you?'}]
    load_pre_validated_pdf('merged_files.pdf')

uploaded_file = st.file_uploader("Choose a document to upload")
if uploaded_file is not None:
    pdf_to_vectordb(uploaded_file)
    comparison = compare_uploaded_pdf_with_context(uploaded_file)
    st.write(f"Similarity with the validated document: {comparison:.2f}%")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = st.session_state.chat.send_message(prompt)
    message = response.get('text')

    st.session_state.messages.append({"role": "assistant", "content": message})
    st.chat_message("assistant").write(message)


