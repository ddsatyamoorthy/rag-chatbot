from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import streamlit as st
from pypdf import PdfReader


def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def create_vectorstore(chunks):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Vector store creation failed: {str(e)}")
        raise

def load_llm():
    return ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

def get_answer(query, vectorstore, llm):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
        response = llm.invoke(prompt)
        return response.content, docs
    except Exception as e:
        st.error(f"❌ Answer generation failed: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
    st.title("📄 RAG PDF Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔍 Processing document..."):
                chunks = process_pdf("temp.pdf")
                if not chunks:
                    st.error("❌ No content found in the document.")
                    return

                vectorstore = create_vectorstore(chunks)
                llm = load_llm()

            st.success("✅ Document ready for queries!")

            query = st.text_input("💬 Ask a question about the document:")
            if query:
                answer, docs = get_answer(query, vectorstore, llm)
                st.markdown("### Answer")
                st.write(answer)

                with st.expander("🔍 Relevant context"):
                    for doc in docs:
                        st.markdown(doc.page_content[:500] + "...")
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main()
