import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ────────────────────────────────────────────────
# 1. Vector store loading
# ────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "mysoft_faiss_index"
    
    if not os.path.exists(index_path):
        st.error(f"FAISS index folder '{index_path}' not found.")
        st.error("Make sure the folder exists in the root of your GitHub repository.")
        st.stop()
    
    try:
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Could not load FAISS index: {str(e)}")
        st.stop()

vector_store = load_vectorstore()

# ────────────────────────────────────────────────
# 2. LLM initialization
# ────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets.")
        st.info("Add it in Streamlit Cloud → Settings → Secrets")
        st.stop()

    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=hf_token,
            task="text-generation",
            temperature=0.15,
            max_new_tokens=600,
            return_full_text=False,
            timeout=180,
        )
        return llm
    except Exception as e:
        st.error(f"LLM initialization failed: {str(e)}")
        st.info("Common fixes:\n• Regenerate token with write role\n• Try repo_id = 'Qwen/Qwen2.5-7B-Instruct'\n• Update langchain-huggingface package")
        st.stop()

llm = get_llm()

# ────────────────────────────────────────────────
# 3. Prompt & chat history formatting
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
    """
    Correct handler for get_chat_history parameter.
    inputs is a dict: {"chat_history": [HumanMessage, AIMessage, ...]}
    """
    chat_history_list = inputs.get("chat_history", [])
    if not chat_history_list:
        return ""

    lines = []
    for message in chat_history_list:
        role = "User" if message.type == "human" else "Assistant"
        content = message.content.strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines) + "\n\n"

# ────────────────────────────────────────────────
# 4. RAG chain
# ────────────────────────────────────────────────
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    get_chat_history=format_chat_history
)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="Mysoft Heaven AI", layout="wide")

st.title("Mysoft Heaven (BD) Ltd. AI Assistant")
st.caption("Answers based only on company documents")

# Chat history persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask about Mysoft Heaven (BD) Ltd...."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            answer = "Sorry — generation failed. Please try again."
            sources = []
            try:
                result = qa_chain.invoke({"question": user_input})
                answer = result["answer"].strip()
                sources = result.get("source_documents", [])
            except Exception as e:
                st.error("Error during generation:")
                st.error(str(e))
                if "input" in str(e).lower() or "key" in str(e).lower():
                    st.info("Input formatting issue – check prompt variables and get_chat_history function.")
                if "410" in str(e) or "Gone" in str(e):
                    st.info("Hugging Face API routing changed – try repo_id = 'Qwen/Qwen2.5-7B-Instruct'")

            st.markdown(answer)

            if sources:
                with st.expander("Reference excerpts"):
                    for i, doc in enumerate(sources, 1):
                        text = doc.page_content.strip()
                        st.markdown(f"**Excerpt {i}**  \n{text[:500]}{'...' if len(text) > 500 else ''}")

    # Save only successful answers
    if "failed" not in answer.lower():
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear button
if st.button("Clear conversation"):
    st.session_state.messages = []
    memory.clear()
    st.rerun()
