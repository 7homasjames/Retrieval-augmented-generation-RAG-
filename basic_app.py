import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import PyPDF2
import textwrap
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

# Import functions from cdb.py
from cdb import upsert_documents, query_collection

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

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        pdf_text = ''
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            pdf_text += page.extract_text() or ''
        
        # Process the text into chunks
        chunk_size = 1000
        chunks = textwrap.wrap(pdf_text, chunk_size)
        all_chunks.extend(chunks)
    
    st.write(f"Total extracted text length: {sum(len(chunk) for chunk in all_chunks)} characters")
    st.write(f"Total number of chunks: {len(all_chunks)}")

    # Upsert chunks into ChromaDB and get their IDs
    ids = upsert_documents(documents=all_chunks)
    st.write(all_chunks)
    print(all_chunks)


    def augment_prompt(query: str):
        query_results = query_collection(query_texts=[query], n_results=3)
        #st.write("Query Results: ", query_results)  # Debug: print the results to Streamlit UI

        # Ensure correct access to documents in query_results
        try:
            source_knowledge = "\n".join([doc for result in query_results['documents'] for doc in result])
        except KeyError as e:
            st.error(f"KeyError: {e}")
            source_knowledge = "No context found due to error."

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
