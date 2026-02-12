import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ────────────────────────────────────────────────
# 1. Load vector store
# ────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "mysoft_faiss_index"
    if not os.path.exists(index_path):
        st.error(f"FAISS index folder '{index_path}' not found. Upload it to your repo.")
        st.stop()
    try:
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load FAISS index: {str(e)}")
        st.stop()

vector_store = load_vectorstore()

# ────────────────────────────────────────────────
# 2. LLM setup (plain Endpoint – avoids chat-completions routing issues)
# ────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN missing in Streamlit secrets.")
        st.info("Add it in app settings → Secrets: HUGGINGFACEHUB_API_TOKEN = \"hf_xxxxxxxxxx\"")
        st.stop()

    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # good free-tier support
            # Alternative: "Qwen/Qwen2.5-7B-Instruct" if rate-limited
            huggingfacehub_api_token=hf_token,
            task="text-generation",
            temperature=0.15,
            max_new_tokens=600,
            return_full_text=False,
            timeout=180,
        )
        return llm
    except Exception as e:
        st.error(f"LLM failed: {str(e)}")
        if "410" in str(e) or "Gone" in str(e):
            st.info("HF API routing issue → try repo_id='Qwen/Qwen2.5-7B-Instruct'")
        st.stop()

llm = get_llm()

# ────────────────────────────────────────────────
# 3. Custom prompt + history formatter
# ────────────────────────────────────────────────
prompt_template = """\
You are a strict company information assistant for **Mysoft Heaven (BD) Ltd.** only.

Rules you MUST follow:
- Answer using ONLY the provided context.
- Do NOT use general knowledge or invent facts.
- If question unrelated to Mysoft Heaven (BD) Ltd. or context insufficient → reply ONLY with:
  "Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Previous conversation:
{chat_history_text}

Relevant context:
{context}

Question: {question}

Concise, professional answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history_text"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def format_chat_history(inputs):
    """Format chat history as text string for the prompt"""
    chat_history = inputs.get("chat_history", [])
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        if isinstance(msg, dict):  # sometimes history is dict-like
            role = msg.get("role", "Unknown")
            content = msg.get("content", "")
        else:  # HumanMessage / AIMessage objects
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content
        lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines) + "\n\n"

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    # Critical fix: provide your custom history variable
    get_chat_history=format_chat_history
)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="Mysoft Heaven AI", layout="wide")

st.title("Mysoft Heaven (BD) Ltd. AI Assistant")
st.caption("Company information only – grounded answers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about Mysoft Heaven (BD) Ltd...."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            answer = "Sorry, generation failed. Please try again."
            sources = []
            try:
                # Invoke with just question – chain handles history internally
                result = qa_chain.invoke({"question": user_input})
                answer = result["answer"].strip()
                sources = result.get("source_documents", [])
            except Exception as e:
                st.error("Generation error:")
                st.error(str(e))
                if "input keys" in str(e).lower():
                    st.info("Input key mismatch – check prompt variables vs chain config.")
                if "410" in str(e) or "Gone" in str(e):
                    st.info("HF changed API path – try different repo_id like 'Qwen/Qwen2.5-7B-Instruct'")

            st.markdown(answer)

            if sources:
                with st.expander("Reference excerpts"):
                    for i, doc in enumerate(sources, 1):
                        text = doc.page_content.strip()
                        st.markdown(f"**Excerpt {i}**  \n{text[:500]}{'...' if len(text) > 500 else ''}")

    # Only save if successful
    if "failed" not in answer.lower():
        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("Clear conversation"):
    st.session_state.messages = []
    memory.clear()
    st.rerun()
