import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# STEP 1: Load & split PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(pages)

# STEP 2: Create vectorstore using HuggingFace embeddings + FAISS
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# STEP 3: Load lightweight open-source model
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# STEP 4: Ask question
def ask_question(vectorstore, llm, question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer based on context:\n\n{context}\n\nQuestion: {question}"
    response = llm(prompt, max_length=300, do_sample=False)[0]['generated_text']
    return response, docs

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG PDF Chatbot")
    st.title("📄 RAG-based PDF Chatbot")
    st.markdown("Upload a PDF and ask questions from it!")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Reading & indexing document..."):
            chunks = process_pdf("temp.pdf")
            vectorstore = create_vectorstore(chunks)
            llm = load_llm()

        st.success("✅ Document processed successfully. Ask your questions below!")

        question = st.text_input("❓ Ask a question")
        if question:
            with st.spinner("🤖 Generating answer..."):
                response, docs = ask_question(vectorstore, llm, question)
                st.markdown("### 📌 Answer")
                st.write(response)
                with st.expander("📚 Context"):
                    for doc in docs:
                        st.text(doc.page_content[:500])

if __name__ == "__main__":
    main()
