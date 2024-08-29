import streamlit as st
from openai_helper import OpenAIChat
import pdfplumber
import uuid


def pdf_to_text(pdf_document):
    doc_id = pdf_document.file_id
    text = ''

    with pdfplumber.open(pdf_document) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return {'id': doc_id, 'text': text}


def pdf_to_vectordb(pdf_document):
    pdf = pdf_to_text(pdf_document)
    text_id = pdf.get('id', uuid.uuid4())
    text = pdf.get('text', '')
    metadata = {"source": "example_source"}
    st.session_state.chat.store_embedding(text, text_id, metadata)


def add_pdf(pdf_document):
    pdf = pdf_to_text(pdf_document)
    text_id = pdf.get('id', uuid.uuid4())
    text = pdf.get('text', '')
    metadata = {"source": "example_source"}
    st.session_state.chat.store_embedding(text, text_id, metadata)


st.title("💬 Chatbot")

if 'chat' not in st.session_state:
    st.session_state.chat = OpenAIChat(openai_model='gpt-4o-mini')
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

uploaded_file = st.file_uploader("Choose a document to upload")
if uploaded_file is not None:
    add_pdf(uploaded_file)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = st.session_state.chat.send_message(prompt)
    message = response.get('text')

    st.session_state.messages.append({"role": "assistant", "content": message})
    st.chat_message("assistant").write(message)


