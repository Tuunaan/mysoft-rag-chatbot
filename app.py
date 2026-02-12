import streamlit as st
import os
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from huggingface_hub import InferenceClient

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
#                   Load LLM Client (Direct HF)
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_token
)

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
#                   Custom RAG Function
# ────────────────────────────────────────────────

@st.cache_data
def get_relevant_docs(query, k=5):
    """Retrieve relevant documents from FAISS"""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

def format_context(docs):
    """Format documents into context string"""
    context = ""
    for i, doc in enumerate(docs):
        context += f"SECTION {i+1}:\n{doc.page_content}\n\n"
    return context

def generate_response(query: str) -> tuple[str, list]:
    """Full RAG pipeline without LangChain chains"""
    # Retrieve relevant docs
    docs = get_relevant_docs(query)
    context = format_context(docs)
    
    # Format full prompt
    full_prompt = prompt_template.format(context=context, question=query)
    
    # Generate with HF InferenceClient
    try:
        response = client.text_generation(
            full_prompt,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.03,
            stop_sequences=["</s>", "Human:", "[INST]"]
        )
        return response, docs
    except Exception as e:
        return f"LLM Error: {str(e)}", docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("Mysoft Heaven AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about Mysoft Heaven"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, source_docs = generate_response(prompt)
            
            st.markdown(answer)
            
            # Show source chunks
            with st.expander("Used document chunks"):
                for i, doc in enumerate(source_docs):
                    st.write(f"**Chunk {i+1}**  \n{doc.page_content[:400]}...")

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
