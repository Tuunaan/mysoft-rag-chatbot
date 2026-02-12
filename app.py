import streamlit as st
import os
from typing import List

from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain_huggingface import HuggingFaceEndpoint

# ────────────────────────────────────────────────
# Config: Use a small, fast, free-tier friendly model
# ────────────────────────────────────────────────
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct"          # Very good 3B model — change to Phi-3.5-mini-instruct or SmolLM3-3B if preferred
# MODEL_REPO = "microsoft/Phi-3.5-mini-instruct"
# MODEL_REPO = "HuggingFaceTB/SmolLM3-3B"

# ────────────────────────────────────────────────
# Load embeddings & vector store (this part is CPU-only & fine)
# ────────────────────────────────────────────────
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vector_store = LangFAISS.load_local(
        "mysoft_faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"Failed to load FAISS index: {str(e)}")
    st.stop()

# ────────────────────────────────────────────────
# LLM via HF router (serverless — no local GPU/RAM needed)
# ────────────────────────────────────────────────
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set in Streamlit secrets.")
    st.stop()

llm = HuggingFaceEndpoint(
    repo_id=MODEL_REPO,
    huggingfacehub_api_token=hf_token,
    # Use router explicitly (fixes old endpoint 410 error)
    endpoint_url="https://router.huggingface.co/hf-inference/models/" + MODEL_REPO,
    temperature=0.15,
    max_new_tokens=400,          # smaller → faster
    top_p=0.9,
)

# ────────────────────────────────────────────────
# Strong grounding prompt
# ────────────────────────────────────────────────
prompt_template = """\
You are a strict company information assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY using the provided context. Do NOT use your general knowledge.
If the question is unrelated or the context lacks enough info, reply only with:

"Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Context:
{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),  # fewer docs → faster
    chain_type_kwargs={"prompt": PROMPT},
    memory=memory,
    return_source_documents=True
)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.title("Mysoft Heaven AI Chatbot")
st.caption("Using small & fast Qwen2.5-3B-Instruct via Hugging Face serverless inference")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Mysoft Heaven..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (may take 5–30 seconds)..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"].strip()

                st.markdown(answer)

                with st.expander("Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Chunk {i+1}**\n{doc.page_content[:350]}...")
            except Exception as e:
                st.error(f"Error: {str(e)}\n\nPossible fixes:\n• Check token in secrets\n• Model may be rate-limited — try later or switch MODEL_REPO")

    st.session_state.messages.append({"role": "assistant", "content": answer})
