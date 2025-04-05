import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# STEP 1: Load and chunk the PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)

# STEP 2: Create vectorstore using HuggingFace embeddings + FAISS
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# STEP 3: Load lightweight open-source model for Streamlit
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# STEP 4: Streamlit UI
def main():
    st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
    st.title("📘 RAG-Based Document Chatbot (Debug Mode Enabled)")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Reading & indexing document..."):
            chunks = process_pdf("temp.pdf")
            vectorstore = create_vectorstore(chunks)
            llm = load_llm()

        st.success("✅ Document processed successfully. Ask your question below.")

        query = st.text_input("❓ Ask a question about the document:")
        if query:
            retriever = vectorstore.as_retriever(search_type="similarity", k=8)  # Increased k for better recall
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Limit token length for model
            MAX_CONTEXT_CHARS = 1500
            short_context = context[:MAX_CONTEXT_CHARS]

            # Add prompt with slight flexibility
            prompt = f"""
You are a helpful assistant. Use only the following context to answer the question. 
If the answer is clearly not present, say "Not found in document."

Context:
{short_context}

Question: {query}

Answer:
"""

            with st.spinner("🧠 Generating answer..."):
                response = llm(prompt, max_new_tokens=256)[0]["generated_text"]
                answer = response.strip()

            st.markdown("### 💡 Answer")
            st.write(answer)

            # Show context for debugging
            with st.expander("🧠 Retrieved Context (for debugging)"):
                st.write(short_context)

            # Show all matched chunks (for validation)
            with st.expander("📄 Top Retrieved Chunks"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(doc.page_content[:500])

if __name__ == "__main__":
    main()
