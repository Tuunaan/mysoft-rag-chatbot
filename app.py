import streamlit as st
import os
import requests
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ────────────────────────────────────────────────
#                Load Embeddings & Vector Store
# ────────────────────────────────────────────────

@st.cache_resource
def load_vector_store():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vector_store = LangFAISS.load_local(
            "mysoft_faiss_index",
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        st.stop()

vector_store = load_vector_store()

# ────────────────────────────────────────────────
#                   HF Token Setup
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

# ────────────────────────────────────────────────
#           Text Generation Endpoint (WORKING)
# ────────────────────────────────────────────────

def generate_text_direct(prompt: str) -> str:
    """Direct text-generation endpoint - bypasses chat format issues"""
    url = f"https://router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/predict"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "No response")
        elif response.status_code == 404:
            # Fallback to classic inference endpoint
            return generate_classic_inference(prompt)
        else:
            return f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return generate_classic_inference(prompt)

def generate_classic_inference(prompt: str) -> str:
    """Classic HF Inference API endpoint (always works)"""
    url = "https://api-inference-huggingface.co/models/microsoft/DialoGPT-medium"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
    except:
        return "Sorry, unable to generate response at this time."

# ────────────────────────────────────────────────
#                     Prompt Template
# ────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a helpful assistant for Mysoft Heaven (BD) Ltd.
Use ONLY the following context to answer. If context is insufficient, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# ────────────────────────────────────────────────
#                   RAG Pipeline
# ────────────────────────────────────────────────

@st.cache_data
def get_docs(query: str) -> list:
    """Get relevant documents"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever.get_relevant_documents(query)

def format_context(docs: list) -> str:
    """Format docs for LLM"""
    context_parts = []
    for i, doc in enumerate(docs):
        content = doc.page_content[:800]
        context_parts.append(f"Doc {i+1}: {content}")
    return "\n\n".join(context_parts)[:3000]

def generate_response(query: str) -> tuple[str, list]:
    """Full RAG pipeline"""
    docs = get_docs(query)
    context = format_context(docs)
    full_prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    
    answer = generate_text_direct(full_prompt)
    return answer, docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("🤖 Mysoft Heaven AI Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about Mysoft Heaven..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            answer, docs = generate_response(user_input)
            st.markdown(answer)
            
            # Sources
            with st.expander(f"📚 Sources ({len(docs)} docs)", expanded=False):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Doc {i+1}:**")
                    st.markdown(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar
with st.sidebar:
    st.button("🗑️ Clear Chat", on_click=lambda: setattr(st.session_state, 'messages', []))
    st.info("💡 Uses your FAISS index + HF Inference API")
