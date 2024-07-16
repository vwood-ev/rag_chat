import streamlit as st
import rag_chat 
import glob
import boto3
import json


## Initialize Bedrock
session = boto3.Session(
    profile_name='137484672202_EVLearningPaths',
    region_name='us-east-1'
)
bedrock = session.client('bedrock-runtime', region_name='us-east-1')


### MainPage

st.set_page_config(page_title="KB Chatbot")
st.title("Knowledge-base Chatbot")

## Options

filenames = glob.glob("data/*")

with st.sidebar:
    debug = st.checkbox("Debug", value=False)
    show_raw = st.checkbox("Show Raw History", value=False)

if 'memory' not in st.session_state:
    st.session_state.memory = rag_chat.get_memory()

if 'chat_history' not in st.session_state:
    # initialize the chat history
    st.session_state.chat_history = []

if 'vector_index' not in st.session_state: 
    with st.spinner("Indexing document..."): 
        st.session_state.vector_index = rag_chat.get_index(bedrock)

print(dir(st.session_state.vector_index))

if show_raw:
    for message in st.session_state.chat_history:
        st.text_area("", message)
else:        
    # Re-render the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
        if debug:
            st.markdown(message["sources"])
            
input_text = st.chat_input("Chat here")

# run the code in this if block after the user submits a chat message
if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    
    st.session_state.chat_history.append({"role":"user", "text":input_text})

    chat_response = rag_chat.get_rag_chat_response(
        client=bedrock, input_text=input_text,
        memory=st.session_state.memory, index=st.session_state.vector_index,
    )
    
    with st.chat_message("assistant"):
        st.markdown(chat_response['answer'])
    
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "text": chat_response['answer'],
            "sources": chat_response['source_documents']
        }
    )





    
