import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import HuggingFaceChat

# ────────────────────────────────────────────────
# 1. Load embeddings & vector store
# ────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.load_local(
            "mysoft_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Failed to load FAISS index: {str(e)}")
        st.stop()

vector_store = load_vectorstore()

# ────────────────────────────────────────────────
# 2. LLM setup using HuggingFaceChat
# ────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets.")
        st.stop()

    try:
        llm = HuggingFaceChat(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  # reliable free-tier model
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            temperature=0.15,
            max_new_tokens=600
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize HuggingFace model: {str(e)}")
        st.stop()

llm = get_llm()

# ────────────────────────────────────────────────
# 3. Strict RAG prompt (forces grounding)
# ────────────────────────────────────────────────
prompt_template = """\
You are a strict company information assistant for **Mysoft Heaven (BD) Ltd.** only.

Rules you MUST follow:
- Answer using **only** the provided context below.
- Do NOT use your general knowledge or make up information.
- If the question is unrelated to Mysoft Heaven (BD) Ltd., or if the context does not contain the answer, reply **only** with this exact sentence:
  "Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Conversation history:
{chat_history}

Current context:
{context}

Question: {question}

Concise, professional answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

# ────────────────────────────────────────────────
# 4. Create conversational RAG chain
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
# 5. Streamlit UI
# ────────────────────────────────────────────────
st.title("Mysoft Heaven AI Assistant")
st.caption("Ask questions about Mysoft Heaven (BD) Ltd. only")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the company..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response safely
    answer = ""
    with st.chat_message("assistant"):
        with st.spinner("Searching company information..."):
            try:
                result = qa_chain.invoke({"question": user_input})
                answer = result["answer"].strip()
                sources = result["source_documents"]

                st.markdown(answer)

                # Optional: show sources
                if sources:
                    with st.expander("Reference chunks", expanded=False):
                        for i, doc in enumerate(sources, 1):
                            content = doc.page_content.strip()
                            st.markdown(f"**Chunk {i}**  \n{content[:450]}{'...' if len(content) > 450 else ''}")

            except Exception as e:
                answer = "Sorry, I am currently unable to process your request. Please try again later."
                st.error(f"Error during generation.")
                st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
