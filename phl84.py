import streamlit as st
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
  

def process_files(files):
    text = ''

    for file in files:
        pdf = PdfReader(file)
        
        for page in pdf.pages:
            text += page.extract_text() 

    return text


def create_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size= 6000,
        chunk_overlap= 1000,
        length_function= len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):  
    embedding_function = HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
    vectorstore = FAISS.from_texts(texts=chunks, embedding= embedding_function)
    vectorstore.save_local("vectorstore")
    return vectorstore

def create_conversation_chain(vectorstore=None):
    if (not vectorstore):
        vectorstore = FAISS.load_local("vectorstore", HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1"), allow_dangerous_deserialization=True)
    llm = HuggingFaceHub(repo_id="HuggingFaceTB/SmolLM3-3B", model_kwargs={"temperature":0.6}, task="conversational")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Converse com seus arquivos", page_icon=":books:")

    if ("conversation" not in st.session_state):
        st.session_state.conversation = None

    st.header("Seus arquivos PDF")
    user_question = st.text_input("Digite sua pergunta sobre o(s) PDF(s) carregados:")
    
    if user_question:
        try:
            response = st.session_state.conversation(user_question)["chat_history"][-1]

            for i, text_message in enumerate(response):

                if (i % 2==0):
                    st.chat_message(text_message.content, is_user=True, key= str(i) + "_user")

                else:
                    st.chat_message(text_message.content, is_user=False, key= str(i) + "_bot")
        except:
            vectorstore = FAISS.load_local("vectorstore", HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1"), allow_dangerous_deserialization=True)
            st.session_state.conversation = create_conversation_chain(vectorstore)
            response = st.session_state.conversation(user_question)["chat_history"][-1]
            for i, text_message in enumerate(response):
                if (i % 2==0):
                    st.chat_message(text_message.content, is_user=True, key= str(i) + "_user")
                else:
                    st.chat_message(text_message.content, is_user=False, key= str(i) + "_bot")

    with st.sidebar:
        st.subheader("Selecione um arquivo ou mais arquivos para conversar")
        pdf_docs = st.file_uploader("Carregue seu pdf abaixo:", type=["pdf"], accept_multiple_files=True)

        if st.button("Processar PDF(s)"):
            if pdf_docs:
                all_files_text = process_files(pdf_docs)

                chunks = create_text_chunks(all_files_text)
                
                vectorstore = create_vector_store(chunks)

                st.session_state.conversation = create_conversation_chain(vectorstore)

                for pdf in pdf_docs:
                    st.session_state[pdf.name] = pdf.read()
                    st.success("PDFs processados com sucesso!")
            else:
                st.error("Por favor, carregue pelo menos um arquivo PDF.")


if __name__ == "__main__":
    main()