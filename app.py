#RAG based MCQ generator project.

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

st.title("PDF-Based MCQ and Q&A Generator")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split the PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # Embedding
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    llm = AzureChatOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.5,
        max_tokens=500
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Prompt templates
    mcq_prompt = """
    Based on the following content, generate 5 multiple choice questions with 4 options each.
    Also, mark the correct option for each question.
    Content:
    {context}
    """

    short_qna_prompt = """
    Based on the following content, generate 5 short answer questions along with concise answers (2-3 lines).
    Content:
    {context}
    """

    # Extract context
    context = "\n".join([doc.page_content for doc in split_docs[:2]])  # limit for speed

    if st.button("Generate MCQs"):
        prompt = mcq_prompt.format(context=context)
        with st.spinner("Generating MCQs..."):
            result = llm.predict(prompt)
        st.markdown("### 📝 MCQ Questions")
        st.write(result)

    if st.button("Generate Short QnA"):
        prompt = short_qna_prompt.format(context=context)
        with st.spinner("Generating Short Questions..."):
            result = llm.predict(prompt)
        st.markdown("### ❓ Short Answer Questions")
        st.write(result)