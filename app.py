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
    st.session_state.uploaded_pdf_text = text
    st.session_state.uploaded_pdf_embedding = st.session_state.chat.store_embedding(text, str(text_id), metadata)


def load_pre_validated_pdf(path):
    validated_pdf = pdf_to_text(path)
    text_id = validated_pdf['id']
    text = validated_pdf['text']
    metadata = {'source': 'pre_validated_pdf'}
    st.session_state.validated_pdf_text = text
    st.session_state.validated_pdf_embedding = st.session_state.chat.store_embedding(text, str(text_id), metadata)

def compare_uploaded_pdf_with_context(uploaded_pdf):
    uploaded_pdf_embedding = st.session_state.uploaded_pdf_embedding
    validated_pdf_embedding = st.session_state.validated_pdf_embedding

    # Check if embeddings are available
    if uploaded_pdf_embedding is None or validated_pdf_embedding is None:
        return None

    # Tokenize the documents into sentences or paragraphs
    uploaded_chunks = st.session_state.chat.chunk_text(st.session_state.uploaded_pdf_text)
    validated_chunks = st.session_state.chat.chunk_text(st.session_state.validated_pdf_text)

    # Calculate embeddings for each chunk
    uploaded_embeddings = [st.session_state.chat.create_embeddings(chunk) for chunk in uploaded_chunks]
    validated_embeddings = [st.session_state.chat.create_embeddings(chunk) for chunk in validated_chunks]

    differences = []

    # Compare each uploaded chunk with the closest validated chunk
    for i, uploaded_embedding in enumerate(uploaded_embeddings):
        if uploaded_embedding:
            # Calculate cosine similarity between this uploaded chunk and all validated chunks
            similarities = cosine_similarity([uploaded_embedding], validated_embeddings)
            max_similarity_index = np.argmax(similarities)
            max_similarity = similarities[0][max_similarity_index]

            # Set a similarity threshold (e.g., 0.85 or 85%)
            if max_similarity < 0.85:
                differences.append({
                    "uploaded_chunk": uploaded_chunks[i],
                    "similarity": max_similarity * 100,
                    "most_similar_validated_chunk": validated_chunks[max_similarity_index]
                })

    # Summarize the overall similarity
    overall_similarity = cosine_similarity([validated_pdf_embedding], [uploaded_pdf_embedding])[0][0] * 100

    st.write(f"Overall similarity with the validated document: {overall_similarity:.2f}%")

    # Highlight differences
    if differences:
        st.write(f"Found {len(differences)} significant differences:")
        for diff in differences:
            st.write(f"- Uploaded chunk: {diff['uploaded_chunk']}")
            st.write(f"  Similarity: {diff['similarity']:.2f}%")
            st.write(f"  Most similar validated chunk: {diff['most_similar_validated_chunk']}")
    else:
        st.write("No significant differences found.")

    return overall_similarity


st.title('\U0001F3E1 Chat with OpenAI RAG AI solution using PDF')

if 'chat' not in st.session_state:
    st.session_state.chat = OpenAIChat(openai_model='gpt-4o-mini')
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I help you?'}]
    load_pre_validated_pdf('merged_files.pdf')

uploaded_file = st.file_uploader("Choose a document to upload")
if uploaded_file is not None:
    pdf_to_vectordb(uploaded_file)
    comparison = compare_uploaded_pdf_with_context(uploaded_file)

    if comparison is not None:
        st.write(f"Similarity with the validated document: {comparison:.2f}%")
        if comparison < 75:
            st.write('there is significant difference between documents please review it and try uploading again.')
    else:

        st.write("Could not calculate similarity. Please check the uploaded document and try again.")


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = st.session_state.chat.send_message(prompt)
    message = response.get('text')

    st.session_state.messages.append({"role": "assistant", "content": message})
    st.chat_message("assistant").write(message)


