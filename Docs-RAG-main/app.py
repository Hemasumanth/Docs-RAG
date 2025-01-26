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

