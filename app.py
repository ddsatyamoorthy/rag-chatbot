import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings  # Corrected import
from transformers import pipeline

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(pages)

def create_vectorstore(chunks):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["HF_API_TOKEN"],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device_map="auto"
    )

def get_answer(query, vectorstore, llm):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = llm(
        prompt,
        max_length=500,
        num_return_sequences=1,
        temperature=0.3
    )
    return result[0]["generated_text"], docs

def main():
    st.set_page_config(page_title="📄 RAG PDF Chatbot", layout="wide")
    st.title("📄 RAG PDF Chatbot")
    st.caption("Built with LangChain, HuggingFace, FAISS, and Streamlit")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
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
            
            with st.expander("🔍 Relevant Document Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.info(doc.page_content[:500] + "...")

if __name__ == "__main__":
    main()
