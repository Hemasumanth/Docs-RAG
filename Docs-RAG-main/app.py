# Imports
import streamlit as st
import warnings
from sklearnrag.generate import QueryAgent
from langchain.memory import ConversationBufferMemory

# Configuration
warnings.filterwarnings("ignore")
st.set_page_config(page_title='Sklearn QA Bot', page_icon='📋', layout="wide")

# UI Setup
icon_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/2560px-Scikit_learn_logo_small.svg.png"
st.markdown(f"""
    <h1 style="text-align: center;">
        <img src="{icon_url}" alt="Icon" style="vertical-align: middle; height: 112px; margin-right: 50px;">
        <span style="color: #F7931E; font-family: 'Sans Serif';">{"Scikit-Learn QA Bot"}</span>
    </h1>
""", unsafe_allow_html=True)
st.write("\n")

# Agent Initialization
system_content = """Answer the query purely using the context provided. But, if the question doesn't seem to be related to
                    Scikit-Learn, then do respond with "I'm sorry, I can only help with scikit-learn related queries".
                    For questions related to API reference, first look at the API Reference not the examples in the context.
                    Do not try to make things up. Be succinct."""

agent = QueryAgent(
    embedding_model_name="thenlper/gte-large",
    llm="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_context_length=32768,
    system_content=system_content
)

# Session State Check
if "messages" not in st.session_state:
    st.session_state['messages'] = []

# Display Messages
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to Stream Responses
def stream_responses():
    result = agent(query=st.session_state['messages'][-1]['content'], stream=True, num_chunks=7)
    for content in result['answer']:
        yield content
    yield "\n\nRelated Sources:\n"
    for i, source in enumerate(result['sources']):
        yield f"{i+1}. {source}\n"

# User Input
prompt = st.chat_input("Hi, I'm your AI assistant to help you in answering Scikit-Learn related queries. Ask me anything!")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message = st.write_stream(stream_responses())
        st.session_state['messages'].append({"role": "assistant", "content": message})

