import os
import streamlit as st

# LangChain imports (Modern 0.2.x structure)
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# -------------------------------------------------------
# 🔐 Load HuggingFace API Token from Streamlit Secrets
# -------------------------------------------------------
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("HuggingFace API token not found. Please set it in Streamlit Secrets.")
    st.stop()


# -------------------------------------------------------
# 🤖 Load LLM (HuggingFace Inference API)
# -------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hf_token,
    temperature=0.1,
    max_new_tokens=512,
)


# -------------------------------------------------------
# 📚 Load Embeddings & FAISS Vector Store
# -------------------------------------------------------
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = LangFAISS.load_local(
    "mysoft_faiss_index",
    embeddings_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# -------------------------------------------------------
# 🧠 Strict Company Prompt
# -------------------------------------------------------
system_prompt = """
You are a strict company information assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY using the provided context.
If the question is unrelated to Mysoft Heaven (BD) Ltd., or the context does not contain enough information, reply only with:

"Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# -------------------------------------------------------
# 🔗 Create Retrieval Chain (Modern LangChain 0.2)
# -------------------------------------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)


# -------------------------------------------------------
# 🌐 Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Mysoft Heaven AI Chatbot", page_icon="🤖")
st.title("🤖 Mysoft Heaven AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about Mysoft Heaven"):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"input": user_input})
                answer = result["answer"]
                st.markdown(answer)

            except Exception as e:
                answer = "An error occurred while processing your request."
                st.error(str(e))

    st.session_state.messages.append({"role": "assistant", "content": answer})
