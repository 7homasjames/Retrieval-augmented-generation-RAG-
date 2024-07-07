# Retrieval-augmented-generation-(RAG)

Retrieval-Augmented Generation (RAG) is a framework that enhances the performance of language models by incorporating external knowledge bases or document stores. RAG combines two main components: Retrieval and Generation, to provide more accurate and contextually relevant responses.

### How RAG Works

1. **Input Query**: A user provides an input query or question.
2. **Document Retrieval**: The retrieval component searches a large corpus or knowledge base to find the most relevant documents or passages related to the query.
3. **Contextual Generation**: The generation component uses both the input query and the retrieved documents to generate a coherent and contextually relevant response.

![RAG Workflow](https://github.com/7homasjames/Retrieval-augmented-generation-RAG-/assets/118433299/8d45c66f-9b99-418f-8d2e-6b0df0f60494)


## Into the Code

1. A basic chatbot is implemented in the `Basics/chatbot.ipynb` directory. This chatbot serves as an example of how to integrate RAG into a conversational AI system.
2. A sample application is created in the `Basics/sample_app.py` directory which makes use of the Streamlit web interface to accept the PDF file and then to generate the output.
3. The RAG implementation is demonstrated using the Pinecone vector database in the `Basics/RAG.ipynb` notebook. This notebook outlines the process of setting up a Pinecone index, embedding documents, and performing similarity searches to augment language model responses.
4. The Streamlit application is demonstrated in `app.py`, which converts the RAG model created in `RAG.ipynb` into a Streamlit application using the FAISS database. This application allows users to upload a PDF file and generate responses based on the content of the PDF using the RAG model.

## How to Use the Streamlit Application

### Prerequisites

- Python 3.6 or higher
- `pip` for installing Python packages

### Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/7homasjames/Retrieval-augmented-generation-RAG-.git
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Example


1. Prepare your `.env` file with your API keys:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

2. Run the main script:

    ```bash
    python -m streamlit run app.py   
    ```


## References

- [RAG: Retrieval-Augmented Generation ](https://youtu.be/wUAUdEw5oxM?si=Rk6Jwo1Qm69fpP7V)
- [Pinecone](https://www.pinecone.io/)
