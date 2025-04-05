import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def load_llm():
    return pipeline("text-generation", model="distilgpt2")

def main():
    st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
    st.title("📘 RAG-Based Document Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Reading & indexing document..."):
            chunks = process_pdf("temp.pdf")
            vectorstore = create_vectorstore(chunks)
            llm = load_llm()

        st.success("✅ Document processed! Ask your question below.")
        query = st.text_input("Ask a question about the document:")

        if query:
            retriever = vectorstore.as_retriever(search_type="similarity", k=3)
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 🧠 Trim context for small LLMs like GPT2
            MAX_CONTEXT_CHARS = 1200
            short_context = context[:MAX_CONTEXT_CHARS]

            # 📄 Final prompt
            prompt = f"""
You are a helpful assistant. Use the context below to answer the question.
Only use the info provided. If the answer is not found, say: 'Not found in document.'

Context:
{short_context}

Question: {query}
Answer:"""

            with st.spinner("🧠 Thinking..."):
                response = llm(prompt, max_new_tokens=300)[0]['generated_text']
                answer = response.split("Answer:")[-1].strip()

            st.markdown("### 💡 Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
