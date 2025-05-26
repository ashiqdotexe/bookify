from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time

load_dotenv()
## load the GROQ API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
## If you do not have open AI key use the below Huggingface embedding
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

## Load PineconeApi key
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(api_key=api_key)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question.
    Work as an helpful assistant.
    <context>
    {context}
    <context>
    Question:{input}

    """
)
st.title("RAG-Based Document Retrieval Chatbot")
st.header("Embedding phase")
index_name = st.text_input("Enter Index Name")
if index_name not in pc.list_indexes().names():
    uploaded_file = st.file_uploader("Please upload your PDF", type="pdf")
    if uploaded_file is not None and index_name and st.button("Document Embedding"):
        loading_msg = st.empty()
        loading_msg.markdown("⏳ Processing...")
        if "vectors" not in st.session_state:
            st.session_state.vectors = None
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.loader = PyPDFLoader("temp_uploaded.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.docs = st.session_state.loader.load()  ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        )
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(
            index=index, embedding=st.session_state.embeddings
        )
        uuids = [str(uuid4()) for _ in range(len(st.session_state.final_documents))]
        vector_store.add_documents(
            documents=st.session_state.final_documents, ids=uuids
        )
        st.session_state.vectors = vector_store
        time.sleep(3)
        loading_msg.empty()
        st.success("🎉 Finished!")
        st.success("Vector store is ready and PDF is embedded.")
else:
    st.info("This file has already been uploaded.")


st.header("RAG Document Q&A")

user_prompt = st.text_input("Please Enter Your Query")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response["answer"])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------")
