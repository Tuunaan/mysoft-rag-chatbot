import streamlit as st
import os
import requests
import json
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ────────────────────────────────────────────────
#                Load Embeddings & Vector Store
# ────────────────────────────────────────────────

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vector_store = LangFAISS.load_local(
        "mysoft_faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"Error loading vector store: {str(e)}")
    st.stop()

# ────────────────────────────────────────────────
#              Hugging Face Router API
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Reliable model on current router

def generate_response_from_hf(prompt: str) -> str:
    """Call Hugging Face router directly using OpenAI-compatible endpoint"""
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["</s>", "Human:", "[INST]"]
    }
    
    try:
        with st.spinner("Generating answer..."):
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                return content
            else:
                return "Error: No valid response received from the model."
                
    except requests.exceptions.HTTPError as http_err:
        try:
            error_detail = response.json()
            error_msg = error_detail.get("error", {}).get("message", str(http_err))
        except:
            error_msg = str(http_err)
        return f"HTTP Error: {error_msg}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# ────────────────────────────────────────────────
#                     Prompt Template
# ────────────────────────────────────────────────

prompt_template = """You are a strict company information assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY using the provided context. Do NOT use your general knowledge.
If the question is unrelated to Mysoft Heaven (BD) Ltd., or if the context does not contain enough information to answer, reply only with:
"Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Context: {context}
Question: {question}
Helpful Answer:"""

# ────────────────────────────────────────────────
#                   RAG Helpers
# ────────────────────────────────────────────────

@st.cache_data
def get_relevant_docs(query, k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

def format_context(docs):
    if not docs:
        return "No relevant information found in the company documents."
    context = ""
    for i, doc in enumerate(docs):
        context += f"SECTION {i+1}:\n{doc.page_content}\n\n"
    return context

def generate_rag_response(query: str) -> tuple[str, list]:
    docs = get_relevant_docs(query)
    context = format_context(docs)
    full_prompt = prompt_template.format(context=context, question=query)
    
    answer = generate_response_from_hf(full_prompt)
    return answer, docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("Mysoft Heaven AI Chatbot")

# Sidebar with Clear Chat button
with st.sidebar:
    st.title("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Mysoft Heaven (BD) Ltd..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        answer, source_docs = generate_rag_response(prompt)
        st.markdown(answer)
        
        # Show sources
        with st.expander("Used document chunks"):
            if not source_docs:
                st.info("No relevant documents were retrieved.")
            else:
                for i, doc in enumerate(source_docs):
                    content_preview = doc.page_content[:380] + "..." if len(doc.page_content) > 380 else doc.page_content
                    st.markdown(f"**Chunk {i+1}**  \n{content_preview}")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
