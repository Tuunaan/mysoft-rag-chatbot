import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ────────────────────────────────────────────────
# 1. Vector store (unchanged)
# ────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "mysoft_faiss_index"
    if not os.path.exists(index_path):
        st.error(f"FAISS index folder '{index_path}' not found.")
        st.stop()
    try:
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load FAISS: {str(e)}")
        st.stop()

vector_store = load_vectorstore()

# ────────────────────────────────────────────────
# 2. LLM – FIXED version
# ────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN missing in secrets.")
        st.stop()

    try:
        endpoint = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=hf_token,
            task="text-generation",
            temperature=0.15,
            max_new_tokens=600,
            return_full_text=False,               # ← this line fixes the validation error
            timeout=180,
        )
        return ChatHuggingFace(llm=endpoint)
    except Exception as e:
        st.error(f"LLM failed: {str(e)}")
        st.stop()

llm = get_llm()

# ────────────────────────────────────────────────
# 3. Prompt (unchanged)
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
# 4. Chain (unchanged)
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
# Streamlit UI (unchanged from your last version)
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

    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("Clear Chat"):
    st.session_state.messages = []
    memory.clear()
    st.rerun()
