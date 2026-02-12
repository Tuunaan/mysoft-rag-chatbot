# Mysoft Heaven AI Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with Streamlit that answers questions **strictly based on company documents** of **Mysoft Heaven (BD) Ltd.**  
It refuses to answer questions outside the provided context or using general knowledge.

## Features

- Document-based answers only (no hallucination)
- Strict instruction-following prompt engineering
- Retrieval using **FAISS** vector store + **all-MiniLM-L6-v2** embeddings
- Generation powered by **Qwen/Qwen2.5-7B-Instruct** via Hugging Face Inference Router
- Modern OpenAI-compatible API endpoint (`router.huggingface.co`)
- Chat history persistence using Streamlit session state
- Source document chunks displayed in expandable section for transparency
- Clear chat history button
- Error handling for missing API token and generation failures


## Tech Stack

- **Frontend / UI**: Streamlit
- **Vector Database**: FAISS (via LangChain)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Qwen/Qwen2.5-7B-Instruct (Hugging Face Inference)
- **HTTP Client**: requests
- **Environment**: Python 3.8+


## priject link
https://mysoft-rag-chatbot-yn4r6wabvvrrjtu7ufj6jn.streamlit.app/
