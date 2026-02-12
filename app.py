import streamlit as st
import os
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ────────────────────────────────────────────────
#                Load Vector Store (Working)
# ────────────────────────────────────────────────

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vector_store = LangFAISS.load_local(
            "mysoft_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        st.stop()

vector_store = load_vector_store()

# ────────────────────────────────────────────────
#           OFFLINE RAG - No External APIs
# ────────────────────────────────────────────────

def generate_offline_response(query: str) -> tuple[str, list]:
    """Pure offline RAG using longest matching chunks"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    
    # Simple rule-based extraction from best document
    best_doc = max(docs, key=lambda d: len(d.page_content))
    context = best_doc.page_content[:2000]
    
    # Extract sentences containing query keywords
    query_words = query.lower().split()
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in query_words):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        answer = " ".join(relevant_sentences[:3]) + "..."
    elif len(context) > 100:
        answer = context[:500] + "..."
    else:
        answer = "No specific information found in company documents for this query."
    
    return answer, docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("🤖 Mysoft Heaven AI Assistant")
st.markdown("*Offline RAG - No external APIs required*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about Mysoft Heaven..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, docs = generate_offline_response(user_input)
            st.markdown(answer)
            
            # Sources
            with st.expander(f"📚 Sources ({len(docs)} documents)", expanded=True):
                for i, doc in enumerate(docs):
                    with st.container():
                        st.markdown(f"**Document {i+1}:**")
                        st.markdown(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar controls
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.success("✅ **Fully Offline** - Uses your FAISS index only")
    st.info("No HF token, no internet required")
