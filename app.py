import streamlit as st
import os
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import requests
import json

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
#              Direct HTTP Client (FIXED)
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

def generate_with_direct_api(prompt: str) -> str:
    """Use Hugging Face router with OpenAI-compatible /v1/chat/completions endpoint"""
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9,
        "frequency_penalty": 1.03,  # similar effect to repetition_penalty
        "stop": ["</s>", "Human:", "[INST]"]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=45)
        response.raise_for_status()
        result = response.json()
        
        # Extract the generated content
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "Error: No content returned from API"
            
    except requests.exceptions.HTTPError as http_err:
        error_detail = ""
        try:
            error_detail = response.text
        except:
            pass
        return f"HTTP Error: {http_err} - {error_detail}"
    except Exception as e:
        return f"API Error: {str(e)}"

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
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

def format_context(docs):
    context = ""
    for i, doc in enumerate(docs):
        context += f"SECTION {i+1}:\n{doc.page_content}\n\n"
    return context

def generate_response(query: str) -> tuple[str, list]:
    docs = get_relevant_docs(query)
    context = format_context(docs)
    full_prompt = prompt_template.format(context=context, question=query)
    
    answer = generate_with_direct_api(full_prompt)
    return answer, docs

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("Mysoft Heaven AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Mysoft Heaven"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, source_docs = generate_response(prompt)
            st.markdown(answer)
            
            with st.expander("Used document chunks"):
                for i, doc in enumerate(source_docs):
                    st.write(f"**Chunk {i+1}**  \n{doc.page_content[:400]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
