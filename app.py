# import streamlit as st
# from langchain.llms import Ollama
# from langchain.vectorstores import FAISS as LangFAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
import streamlit as st

# from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load embeddings and vector store
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = LangFAISS.load_local(
    "mysoft_faiss_index",
    embeddings_model,
    allow_dangerous_deserialization=True
)
# llm = Ollama(model="mistral")
# ///////////////////////////////////////////////////////////////////////////////////
from langchain_huggingface import HuggingFaceEndpoint
import os

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hf_token,
    temperature=0.1,
    max_new_tokens=512
)
# ///////////////////////////////////////////////////////////////////////////////////
# ────────────────────────────────────────────────
# Strong prompt that enforces grounding + rejection
# ────────────────────────────────────────────────
prompt_template = """\
You are a strict company information assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY using the provided context. Do NOT use your general knowledge.
If the question is unrelated to Mysoft Heaven (BD) Ltd., or if the context does not contain enough information to answer, reply only with:

"Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Context:
{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

st.title("Mysoft Heaven AI Chatbot")

# Simple chat-like interface
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
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                st.markdown(answer)

                # Optional: show sources for debugging / demo
                with st.expander("Used document chunks"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Chunk {i+1}**  \n{doc.page_content[:400]}...")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running (`ollama run mistral`)")


    st.session_state.messages.append({"role": "assistant", "content": answer})
