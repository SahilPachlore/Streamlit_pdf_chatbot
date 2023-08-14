from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_extras.add_vertical_space import add_vertical_space
import os
# Sidebar contents
with st.sidebar:
    st.title('SAHIL CHATBot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Sahil Pachlore](https://sahilpachlore.com)')

load_dotenv()
def main():
     st.header("Chat with PDF 💬")
     

     pdf = st.file_uploader("Upload your PDF", type='pdf')
   

     if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        # # # embeddings
  

 
        
       
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)            
            with open(f"{store_name}.pkl", "wb") as f:
              pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        
        if query:
            # Define your prompt
            prompt = "You are a knowledgeable  AI assistant. Please provide detailed answers to the following questions in very Lord Krishna way like very polite:\n"
            
            # Concatenate the prompt and query
            input_text = prompt + query

            docs = VectorStore.similarity_search(query=input_text, k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=input_text)
                print(cb)
            st.write(response)
if __name__ == '__main__':
    main()    