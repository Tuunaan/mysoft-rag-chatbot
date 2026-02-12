import streamlit as st
import os
import requests
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
#                   HF Token Setup
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

# ────────────────────────────────────────────────
#              OpenAI-Compatible Chat Endpoint
# ────────────────────────────────────────────────

def generate_with_hf_chat(prompt: str) -> str:
    """Direct HTTP to HF Router Chat Completions (most reliable)"""
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback model if Mistral fails
        return generate_fallback(prompt, str(e))

def generate_fallback(prompt: str, error: str) -> str:
    """Fallback to a reliable free model"""
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": "microsoft/DialoGPT-medium",  # Free, reliable fallback
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.1
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except:
        return f"Sorry, service temporarily unavailable: {error[:100]}..."

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
#                   RAG Pipeline
# ────────────────────────────────────────────────

@st.cache_data
def get_relevant_docs(_query: str, k: int = 5):
    """Retrieve relevant documents from FAISS"""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(_query)
    return docs

def format_context(docs: list) -> str:
    """Format documents into context string"""
    context = ""
    for i, doc in enumerate(docs):
        context += f"SECTION {i+1}:\n{doc.page_content[:1000]}\n\n"
    return context[:4000]  # Truncate to avoid token limits

def generate_response(query: str) -> tuple[str, list]:
    """Complete RAG pipeline"""
    docs = get_relevant_docs(query)
    context = format_context(docs)
    full_prompt = prompt_template.format(context=context, question=query)
    
    answer = generate_with_hf_chat(full_prompt)
    return answer, docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("🤖 Mysoft Heaven AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Mysoft Heaven (BD) Ltd..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching company knowledge base..."):
            answer, source_docs = generate_response(prompt)
            st.markdown(answer)
            
            # Source documents expander
            with st.expander(f"📄 Sources ({len(source_docs)} documents)", expanded=False):
                for i, doc in enumerate(source_docs):
                    with st.container():
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
