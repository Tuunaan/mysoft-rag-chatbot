import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ────────────────────────────────────────────────
# 1. Load vector store (cached for speed on Streamlit Cloud)
# ────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "mysoft_faiss_index"
    if not os.path.exists(index_path):
        st.error(f"FAISS index folder '{index_path}' not found. Please upload it to your repo.")
        st.stop()
    try:
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Failed to load FAISS index: {str(e)}")
        st.stop()

vector_store = load_vectorstore()

# ────────────────────────────────────────────────
# 2. LLM setup (serverless, with fallback & debug info)
# ────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets.")
        st.info("→ Go to app settings → Secrets → add: HUGGINGFACEHUB_API_TOKEN = \"hf_xxxxxxxxxx\"")
        st.stop()

    try:
        endpoint = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",           # reliable free serverless model
            # repo_id="mistralai/Mistral-7B-Instruct-v0.3",   # alternative – may need task="conversational"
            huggingfacehub_api_token=hf_token,
            task="text-generation",                           # ← prevents dedicated endpoint routing
            temperature=0.15,
            max_new_tokens=600,
            model_kwargs={"return_full_text": False},
            timeout=180,
        )
        return ChatHuggingFace(llm=endpoint)
    except Exception as e:
        err_str = str(e).lower()
        st.error(f"LLM initialization failed: {str(e)}")
        if "403" in err_str or "inference.endpoints.read" in err_str:
            st.warning("""
            This error usually means LangChain tried to use **dedicated Inference Endpoints** instead of free serverless.
            Fixes to try:
            1. Regenerate token at https://huggingface.co/settings/tokens → choose **write** role
            2. Try repo_id = "Qwen/Qwen2.5-7B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.3"
            3. Update packages: add to requirements.txt → langchain-huggingface>=0.1.0 (or latest)
            """)
        elif "provider" in err_str or "stopiteration" in err_str:
            st.info("Model may not be available on any free provider right now → try different repo_id")
        st.stop()

llm = get_llm()

# ────────────────────────────────────────────────
# 3. Strict prompt – company info only + history
# ────────────────────────────────────────────────
prompt_template = """\
You are a strict company information assistant for **Mysoft Heaven (BD) Ltd.** only.

Rules (must follow exactly):
- Use **only** the provided context to answer.
- Never use your general knowledge or invent facts.
- If the question is unrelated to Mysoft Heaven (BD) Ltd. or the context lacks info, reply **only** with:
  "Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Conversation history:
{chat_history}

Context from documents:
{context}

Question: {question}

Short, professional answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

# ────────────────────────────────────────────────
# 4. RAG chain with memory
# ────────────────────────────────────────────────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.set_page_config(page_title="Mysoft Heaven AI", layout="wide")

st.title("Mysoft Heaven (BD) Ltd. AI Assistant")
st.caption("Company information only – grounded answers")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if user_input := st.chat_input("Ask about Mysoft Heaven (BD) Ltd...."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking company documents..."):
            try:
                result = qa_chain.invoke({"question": user_input})
                answer = result["answer"].strip()
                sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Source excerpts (for reference)"):
                        for i, doc in enumerate(sources, 1):
                            text = doc.page_content.strip()
                            st.markdown(f"**Excerpt {i}**  \n{text[:480]}{'...' if len(text) > 480 else ''}")

            except Exception as e:
                st.error("Generation failed.")
                st.error(str(e))
                if "token" in str(e).lower() or "403" in str(e):
                    st.info("Check your HF token in Streamlit secrets (must have write access)")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    memory.clear()
    st.rerun()
