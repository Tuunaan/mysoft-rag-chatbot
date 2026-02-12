import streamlit as st
import os
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
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
#              Hugging Face Inference Client
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

client = InferenceClient(token=hf_token)

# Use a model with better free/warm support in 2026
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # ← Changed here — good balance of quality & availability

def generate_with_hf_client(prompt: str) -> str:
    """Generate response using Hugging Face InferenceClient"""
    try:
        completion = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "Human:", "[INST]"],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = f"{error_msg} - {json.dumps(error_detail)}"
            except:
                pass
        return f"Generation Error: {error_msg}"

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
    
    answer = generate_with_hf_client(full_prompt)
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
                    if len(doc.page_content) > 400:
                        st.write("...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
