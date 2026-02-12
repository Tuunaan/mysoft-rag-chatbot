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
# Use a small model that IS supported on HF Inference Providers router
# ────────────────────────────────────────────────
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct"  # Reliable small model in 2026
# Alternatives: "microsoft/Phi-3.5-mini-instruct", "google/gemma-2-2b-it"

# ────────────────────────────────────────────────
# Load vector store & embeddings (fine on CPU)
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
# LLM via HF router (serverless, no local GPU/RAM needed)
# ────────────────────────────────────────────────
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("Missing HUGGINGFACEHUB_API_TOKEN in Streamlit secrets.")
    st.stop()

llm = HuggingFaceEndpoint(
    repo_id=MODEL_REPO,
    huggingfacehub_api_token=hf_token,
    temperature=0.15,
    max_new_tokens=400,          # Keep small for speed
    top_p=0.9,
    # No endpoint_url needed — let it auto-resolve via providers
    # If still fails, try: provider="auto" (if your langchain-huggingface version supports it)
)

# ────────────────────────────────────────────────
# Prompt (strict grounding)
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

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": PROMPT},
    memory=memory,
    return_source_documents=True
)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.title("Mysoft Heaven AI Chatbot")
st.caption(f"Powered by {MODEL_REPO} via Hugging Face serverless (responses may take 5–30s)")

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
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"].strip()
                st.markdown(answer)

                with st.expander("Sources (document chunks)"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Chunk {i+1}**  \n{doc.page_content[:350]}...")
            except Exception as e:
                st.error(f"Generation error: {str(e)}\n\nPossible fixes:\n- Check token in secrets\n- Model may be temporarily unavailable → try changing MODEL_REPO\n- Rate limit → wait a few minutes")

    st.session_state.messages.append({"role": "assistant", "content": answer})
