import time
import cv2
import os
import streamlit as st
from llama_index import  ServiceContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
from llama_index import SimpleDirectoryReader
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Chatbot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def page_1():
    #st.header("Chat with your Data")
    col1, col2 = st.columns([0.2, 0.8])
    col1.image('./data/David.jpeg')
    col2.subheader("David says")
       
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the machine learning and your job is to answer technical questions."))
    index  = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query=st.text_input("Ask questions related to your Data")
    with st.spinner('Please wait...'):
        if query:
            chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
            response = chat_engine.chat(query)
            col2.write(response.response)

def page_2():
    st.header("Chat with your Data")
    col1, col2 = st.columns([3, 1])
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the machine learning and your job is to answer technical questions."))
    index  = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query=st.text_input("Ask questions related to your Data")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)

def page_3():
    st.header("Chat with your Data")
    col1, col2 = st.columns([3, 1])
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the machine learning and your job is to answer technical questions."))
    index  = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query=st.text_input("Ask questions related to your Data")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)

#with st.sidebar:
#    st.title("Chat with your Data")
#    run = st.checkbox('Run')
#    FRAME_WINDOW = st.image([])
#    camera = cv2.VideoCapture(0)
#    while run:
#        _, frame = camera.read()
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        FRAME_WINDOW.image(frame)
    #image_holder = st.empty()
    #for item in images:
    #    image_holder.image(item)
    #    time.sleep(5000)

PAGES = {
    "David Attenborough": page_1,
    "Zia Mohyeddin" : page_2,
    "Amitabh Bachchan" : page_3
}

def main():
    st.sidebar.title('Navigation')
    choice=st.sidebar.selectbox('Select personality', list(PAGES.keys()))
    PAGES[choice]()
    
if __name__=='__main__':
    main()    
ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the ML"))