import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from transformers import pipeline

# STEP 1: Load and split PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

# STEP 2: Create vectorstore using Hugging Face API embeddings
def create_vectorstore(chunks):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HF_API_TOKEN"],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

# STEP 3: Load the QA model
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# STEP 4: Answer questions using context and model
def get_answer(query, vectorstore, llm):
    docs = vectorstore.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = llm(prompt, max_length=300, do_sample=False)
    return result[0]["generated_text"], docs

# STREAMLIT UI
def main():
    st.set_page_config(page_title="📄 RAG PDF Chatbot", layout="wide")
    st.title("📄 RAG PDF Chatbot")
    st.caption("Built with LangChain, HuggingFace, FAISS, and Streamlit")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Reading & indexing document..."):
            chunks = process_pdf("temp.pdf")
            vectorstore = create_vectorstore(chunks)
            llm = load_llm()

        st.success("✅ Document processed successfully. Ask your questions!")

        query = st.text_input("💬 Ask a question about the document:")
        if query:
            answer, docs = get_answer(query, vectorstore, llm)
            st.markdown("### ✅ Answer")
            st.write(answer)
            with st.expander("🔍 Source Chunks"):
                for doc in docs:
                    st.text(doc.page_content[:500])

if __name__ == "__main__":
    main()
