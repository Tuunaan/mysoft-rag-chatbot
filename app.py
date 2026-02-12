import streamlit as st
import os
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from huggingface_hub import InferenceClient

# ────────────────────────────────────────────────
#                Load Embeddings & Vector Store
# ────────────────────────────────────────────────

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vector_store = LangFAISS.load_local(
        "mysoft_faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"Error loading vector store: {str(e)}")
    st.stop()

# ────────────────────────────────────────────────
#                   Load LLM Client (Direct HF)
# ────────────────────────────────────────────────

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not set. Please add it in Streamlit secrets.")
    st.stop()

# Direct HuggingFace InferenceClient (bypasses LangChain endpoint issues)
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_token
)

def invoke_llm(prompt: str) -> str:
    """Custom LLM wrapper for RAG chain compatibility"""
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.03,
            stop_sequences=["</s>", "[INST]", "Human:"]
        )
        return response
    except Exception as e:
        return f"LLM Error: {str(e)}"

# ────────────────────────────────────────────────
#                     Prompt Template
# ────────────────────────────────────────────────

prompt_template = """You are a strict company information assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY using the provided context. Do NOT use your general knowledge.
If the question is unrelated to Mysoft Heaven (BD) Ltd., or if the context does not contain enough information to answer, reply only with:
"Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Context: {context}
Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ────────────────────────────────────────────────
#                      Custom Chain
# ────────────────────────────────────────────────

class CustomLLM:
    """Wrapper to make InferenceClient compatible with LangChain"""
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        return invoke_llm(prompt)
    
    @property
    def _llm_type(self) -> str:
        return "custom"

llm = CustomLLM()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT},
    memory=memory,
    return_source_documents=True
)

# ────────────────────────────────────────────────
#                   Streamlit UI
# ────────────────────────────────────────────────

st.title("Mysoft Heaven AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about Mysoft Heaven"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]

                st.markdown(answer)

                # Optional: show source chunks (good for debugging)
                with st.expander("Used document chunks"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Chunk {i+1}**  \n{doc.page_content[:400]}...")

            except Exception as e:
                error_msg = f"Error during generation: {str(e)}"
                st.error(error_msg)
                answer = error_msg

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
