import streamlit as st
import os
from typing import Any, List

from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ────────────────────────────────────────────────
# Load embeddings and vector store
# ────────────────────────────────────────────────
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vector_store = LangFAISS.load_local(
        "mysoft_faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
except Exception as e:
    st.error(f"Failed to load FAISS index: {str(e)}")
    st.stop()

# ────────────────────────────────────────────────
# Load Mistral model locally using Transformers (as per model card recommendation)
# ────────────────────────────────────────────────
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"  # Auto-assign to GPU if available
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.1,
        top_p=0.95,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    st.error(f"Failed to load Mistral model: {str(e)}. Ensure you have sufficient RAM/GPU and the model is downloaded.")
    st.stop()

# ────────────────────────────────────────────────
# Strong grounding prompt (updated to include chat history)
# ────────────────────────────────────────────────
prompt_template = """\
[INST] You are a strict company information assistant for Mysoft Heaven (BD) Ltd.
Answer ONLY using the provided context. Do NOT use your general knowledge.
If the question is unrelated to Mysoft Heaven (BD) Ltd., or if the context does not contain enough information to answer, reply only with:

"Sorry, I can only answer questions about Mysoft Heaven (BD) Ltd. based on the provided company information."

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Helpful Answer: [/INST]"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"]
)

# ────────────────────────────────────────────────
# Memory for conversation history
# ────────────────────────────────────────────────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.title("Mysoft Heaven AI Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question about Mysoft Heaven"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant documents
                docs: List[Document] = retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Get chat history from memory (format as string)
                chat_history = memory.load_memory_variables({})["chat_history"]
                formatted_history = ""
                if isinstance(chat_history, list):
                    for msg in chat_history:
                        if hasattr(msg, 'type') and msg.type == "human":
                            formatted_history += f"User: {msg.content}\n"
                        elif hasattr(msg, 'type') and msg.type == "ai":
                            formatted_history += f"Assistant: {msg.content}\n"

                # Build full prompt
                full_prompt = PROMPT.format(
                    chat_history=formatted_history,
                    context=context,
                    question=user_input
                )

                # Generate answer using the pipeline
                result = llm(full_prompt)

                # Extract the generated text (transformers pipeline returns a list of dicts)
                answer = result if isinstance(result, str) else result[0]['generated_text']

                # Clean up answer: Remove the prompt part if present
                if "[/INST]" in answer:
                    answer = answer.split("[/INST]")[-1].strip()
                if "Helpful Answer:" in answer:
                    answer = answer.split("Helpful Answer:")[-1].strip()

                st.markdown(answer)

                # Save to memory
                memory.save_context(
                    {"input": user_input},
                    {"output": answer}
                )

                # Optional: show sources
                with st.expander("Used document chunks"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Chunk {i+1}**  \n{doc.page_content[:400]}...")

            except Exception as e:
                error_msg = f"Error during generation: {str(e)}"
                st.error(error_msg)
                answer = error_msg

    # Add assistant response to displayed history
    st.session_state.messages.append({"role": "assistant", "content": answer})
