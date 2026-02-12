import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

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
# 2. LLM setup with free-tier compatible model
# ────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN not found in secrets.")
        st.stop()

    try:
        endpoint = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.15,
            max_new_tokens=300,
            timeout=120
        )
        llm = ChatHuggingFace(llm=endpoint)
        return llm
    except Exception as e:
        st.error(f"Failed to initialize HuggingFace model: {str(e)}")
        st.stop()

llm = get_llm()

# ────────────────────────────────────────────────
# 3. Strict RAG prompt
# ────────────────────────────────────────────────
prompt_template = """\
You are a strict company information assistant for **Mysoft Heaven (BD) Ltd.** only.

Rules you MUST follow:
- Answer using **only** the provided context below.
- Do NOT use your general knowledge or make up information.
- If the question is unrelated to Mysoft Heaven (BD) Ltd., or if the context does not contain the answer, reply:
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
# 4. Conversational RAG chain
# ────────────────────────────────────────────────
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the company..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = ""
    with st.chat_message("assistant"):
        with st.spinner("Searching company information..."):
            try:
                result = qa_chain.invoke({"question": user_input})
                answer = result["answer"].strip()
                sources = result["source_documents"]

                st.markdown(answer)

                if sources:
                    with st.expander("Reference chunks"):
                        for i, doc in enumerate(sources, 1):
                            text = doc.page_content.strip()
                            st.markdown(f"**Chunk {i}:** {text[:400]}{'...' if len(text) > 400 else ''}")

            except Exception as e:
                answer = "Sorry, I am currently unable to process your request."
                st.error(f"Error during generation: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
