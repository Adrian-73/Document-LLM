import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub, ctransformers




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_Hub_llm():

    llm = HuggingFaceHub( 
        # repo_id="google/flan-ul2",
        repo_id="HuggingFaceH4/zephyr-7b-beta", 
                         model_kwargs={"temperature":0.1, 
                                       "max_length":2048, 
                                        "top_k":50,
                                        "task":"text-generation",
                                        "num_return_sequences":3,
                                       "top_p":0.95})
    
    return llm

def get_local_llm():
    llm = ctransformers.CTransformers(
        model = "C:/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type = "llama",
        max_new_tokens = 1024,
        max_length = 4096,
        temperature = 0.1
    )
    return llm


def get_conversation_chain(vectorstore,llm):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    if vectorstore:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type = "stuff",
            verbose=True,
            retriever=vectorstore.as_retriever(search_kwargs = {"k" : 3, "search_type" : "similarity"}),
            memory=memory
        )
    else:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type = "stuff",
            verbose=True,
            memory=memory
        )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("User"):
                st.write( message.content)
        else:
            with st.chat_message("assistant"):
                st.write( message.content)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs ")
    
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
            
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create llm
                llm = get_Hub_llm()
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore,llm)

    

if __name__ == '__main__':
    main()