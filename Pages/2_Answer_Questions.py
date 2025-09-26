
import streamlit as st


# --- Custom CSS for chat bubbles and input row ---
st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #dbeafe;
        color: #222;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 6px 0;
        text-align: right;
        font-size: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        width: fit-content;
        float: right;
        max-width: 80%;
    }
    .chat-bubble-assistant {
        background-color: #f3f4f6;
        color: #222;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 6px 0;
        text-align: left;
        font-size: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        width: fit-content;
        float: left;
        max-width: 80%;
    }
    .clearfix { clear: both; }
    .input-row {
        display: flex;
        align-items: center;
        gap: 0;
        margin-top: 1rem;
        margin-bottom: 1rem;
        width: 100%;
    }
    .input-box {
        flex: 1;
        padding: 0.7rem 1rem 0.7rem 1rem;
        font-size: 1rem;
        border-radius: 18px;
        border: 1px solid #d1d5db;
        outline: none;
        background: #fff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .input-wrapper {
        width: 100%;
        display: flex;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Branding ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("images/Deloitte.png", width=85)
with col_title:
    st.markdown("""
        <div style="background-color: #111; color: #fff; padding: 0 20px; height: 52px; display: flex; align-items: center; border-bottom: 2px solid #222; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 0px; border-radius: 18px;">
            <h1 style="color: #fff; margin: 2; font-size: 2rem; font-family: 'Segoe UI', Arial, sans-serif;">
                SEC Filing Chatbot
            </h1>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

import os
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrock

import boto3
from config import *

from dotenv import load_dotenv

load_dotenv()  # Take environment variables from .env.

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
embedding_model = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")
llm_model = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region=AWS_REGION
)

# --- Index selection ---
available_indexes = [d for d in os.listdir(INDEXES_DIR) if os.path.isdir(os.path.join(INDEXES_DIR, d))]
if not available_indexes:
    st.warning("No indexes found. Please create one first.")
    st.stop()

search_mode = st.radio("Search Mode", ["Single Index", "Multi Index"])
if search_mode == "Single Index":
    selected_index = st.selectbox("Choose an index", available_indexes)
    selected_indexes = [selected_index] if selected_index else []
else:
    selected_indexes = st.multiselect("Choose indexes to search", available_indexes)
    if not selected_indexes:
        st.info("Select at least one index to search.")
        st.stop()

if not selected_indexes or any(idx == "" for idx in selected_indexes):
    st.info("Please select index(es) to proceed.")
    st.stop()

# --- Load FAISS indexes ---
faiss_dbs = []
for idx in selected_indexes:
    persist_path = os.path.join(INDEXES_DIR, idx)
    try:
        faiss_db = FAISS.load_local(
            persist_path,
            embedding_model,
            allow_dangerous_deserialization=True  # Only if you trust the source!
        )
        faiss_dbs.append(faiss_db)
    except Exception as e:
        st.error(f"Could not load FAISS index '{idx}': {e}")
        st.stop()

def get_bot_answer(user_query, faiss_dbs):
    top_docs = []
    for faiss_db in faiss_dbs:
        docs = faiss_db.similarity_search(user_query, k=5)
        top_docs.extend(docs)
    context = ""
    for i, doc in enumerate(top_docs, 1):
        context += f"\n\n---\n{doc.page_content}"
    if not context:
        context = "No relevant context found in the selected indexes."
    system_prompt = (
        "You are an expert assistant answering questions based on SEC filings from an index. "
        "Always reply in clear Markdown format. Always reply in the same font structure."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{user_query}\n\nAnswer clearly and concisely. If the context does not contain the info, say so.")
    ]
    answer = llm_model(messages)
    formatted_answer = str(getattr(answer, "content", answer)).strip()
    return formatted_answer

# --- Chat state initialization ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "input_value" not in st.session_state:
    st.session_state["input_value"] = ""
if "awaiting_answer" not in st.session_state:
    st.session_state["awaiting_answer"] = False
if "form_key" not in st.session_state:
    st.session_state["form_key"] = 0

chat_container = st.container()
with chat_container:
    # Display chat history
    for chat in st.session_state["chat_history"]:
        col1, col2 = st.columns([2, 5])
        with col2:
            st.markdown(f'<div class="chat-bubble-user"><b>You:</b> {chat["user"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([5, 2])
        with col1:
            st.markdown(f'<div class="chat-bubble-assistant"><b>Assistant:</b><br>{chat["bot"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)

    # --- Input row: text box only, no send button ---
    form_key = f"chat_form_{st.session_state['form_key']}"
    input_key = f"input_{st.session_state['form_key']}"

    st.markdown('<div class="input-row"><div class="input-wrapper">', unsafe_allow_html=True)
    user_input = st.text_input(
        "",
        value=st.session_state["input_value"],
        key=input_key,
        placeholder="Type your question and press Enter..."
    )
    st.markdown('</div></div>', unsafe_allow_html=True)

    # --- Clear chat button directly below input box ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.session_state["input_value"] = ""
        st.session_state["awaiting_answer"] = False
        st.session_state["form_key"] += 1  # Force form/input reset

    # --- Submission logic ---
    submitted = False
    # Enter key triggers rerun and updates session_state[input_key]
    if user_input and user_input != st.session_state["input_value"]:
        submitted = True
        st.session_state["input_value"] = user_input
        st.session_state["awaiting_answer"] = True

    # --- Process answer if awaiting ---
    if st.session_state["awaiting_answer"]:
        with st.spinner("Assistant is thinking..."):
            bot_response = get_bot_answer(st.session_state["input_value"], faiss_dbs)
        st.session_state["chat_history"].append({
            "user": st.session_state["input_value"],
            "bot": bot_response
        })
        st.session_state["input_value"] = ""  # Clear input after answer is appended
        st.session_state["awaiting_answer"] = False
        st.session_state["form_key"] += 1  # Change form key to reset input
        st.rerun()
