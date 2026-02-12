import streamlit as st
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# ────────────────────────────────────────────────
#                Load Vector Store
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
#           FIXED Offline RAG Logic
# ────────────────────────────────────────────────

def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_relevant_content(query, docs):
    """Extract relevant content using multiple strategies"""
    query_lower = query.lower()
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    best_sentences = []
    best_chunks = []
    
    for doc in docs:
        content = clean_text(doc.page_content)
        content_lower = content.lower()
        
        # Strategy 1: Exact keyword matches
        for word in query_words:
            if word in content_lower and len(word) > 2:
                sentences = re.split(r'[.!?]+', content)
                for sent in sentences:
                    if word in sent.lower() and len(sent.strip()) > 20:
                        best_sentences.append(clean_text(sent))
        
        # Strategy 2: Full chunk if highly relevant
        score = sum(1 for word in query_words if word in content_lower)
        if score >= len(query_words) * 0.5:  # 50% keyword match
            best_chunks.append(content[:800])
    
    # Combine results
    result = []
    if best_sentences:
        result.extend(best_sentences[:4])
    if best_chunks:
        result.append(best_chunks[0])
    
    if not result:
        # Fallback: longest relevant chunk
        result = [max(docs, key=lambda d: len(clean_text(d.page_content).split())).page_content[:800]]
    
    return " ".join(result)[:2500]

def generate_offline_response(query: str) -> tuple[str, list]:
    """Robust offline RAG with better extraction"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    
    context = extract_relevant_content(query, docs)
    
    # Generate structured response
    if "mysoft" in query.lower() or "heaven" in query.lower():
        answer = f"**Mysoft Heaven (BD) Ltd** converts clients' Product Vision into complete product development. The development lifecycle is controlled by client inputs and direction, providing complete independence and flexibility within budget provisions.\n\n**Relevant details from documents:**\n{context[:1000]}..."
    else:
        answer = f"**Answer based on company documents:**\n\n{context[:1200]}...\n\n*This information is extracted directly from Mysoft Heaven (BD) Ltd. company documents.*"
    
    return answer, docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("🤖 Mysoft Heaven AI Assistant")
st.markdown("**Offline RAG Chatbot** - Works with your FAISS index")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about Mysoft Heaven..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching company documents..."):
            answer, docs = generate_offline_response(user_input)
            st.markdown(answer)
            
            # Show sources
            with st.expander(f"📚 Source Documents ({len(docs)} found)", expanded=True):
                for i, doc in enumerate(docs):
                    st.markdown(f"---\n**📄 Document {i+1}:**")
                    preview = doc.page_content[:600]
                    st.markdown(f"`{preview}...`")

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar
with st.sidebar:
    st.button("🗑️ Clear Chat", on_click=lambda: setattr(st.session_state, 'messages', []))
    st.success("✅ **Fully Offline & Working**")
    st.info("🔧 Uses advanced keyword extraction + sentence matching")
