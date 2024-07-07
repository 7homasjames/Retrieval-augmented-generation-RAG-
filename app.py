import os
import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import numpy as np
import PyPDF2
import pandas as pd
import textwrap
import faiss
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Streamlit app
st.title('PDF Question Answering System')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = ''
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        pdf_text += page.extract_text() or ''

    with open('extracted_text.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(pdf_text)
    
    st.write(f"Extracted text length: {len(pdf_text)} characters")

    with open('extracted_text.txt', 'r', encoding='utf-8') as file:
        pdf_text = file.read()

    chunk_size = 1000
    chunks = textwrap.wrap(pdf_text, chunk_size)

    data = pd.DataFrame({
        'text': chunks,
        'doi': ['doi_value']*len(chunks),
        'chunk-id': range(len(chunks)),
        'source': ['source_value']*len(chunks),
        'title': ['title_value']*len(chunks)
    })

    embeddings = embed_model.embed_documents(data['text'].tolist())

    # Initialize FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(np.array(embeddings).astype('float32'))

    def augment_prompt(query: str):
        query_embedding = np.array(embed_model.embed_query(query)).astype('float32')
        _, I = index.search(np.array([query_embedding]), k=3)
        source_knowledge = "\n".join([data.iloc[i]['text'] for i in I[0]])
        augmented_prompt = f"""Using the contexts below, answer the query.
        
        Contexts:
        {source_knowledge}
        
        Query: {query}"""
        return augmented_prompt

    query = st.text_input("Ask a question about the PDF content")

    if query:
        prompt = HumanMessage(content=augment_prompt(query))

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hi AI, how are you today?"),
            AIMessage(content="I'm great thank you. How can I help you?"),
            prompt
        ]

        res = chat(messages)
        st.write(res.content)
