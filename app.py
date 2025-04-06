import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(pages)

def create_vectorstore(chunks):
    try:
        if not st.secrets.get("OPENAI_API_KEY"):
            st.error("❌ Missing OpenAI API key in Streamlit secrets!")
            raise ValueError("Missing OPENAI_API_KEY")
        
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Vector store creation failed: {str(e)}")
        raise

def load_llm():
    return ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )

def get_answer(query, vectorstore, llm):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the following question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = llm.invoke(prompt)
        return response.content, docs
    except Exception as e:
        st.error(f"❌ Answer generation failed: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="📄 RAG PDF Chatbot", layout="wide")
    st.title("📄 RAG PDF Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔍 Processing document..."):
                chunks = process_pdf("temp.pdf")
                if not chunks:
                    st.error("❌ No text extracted from PDF")
                    return

                vectorstore = create_vectorstore(chunks)
                llm = load_llm()

            st.success("✅ Document ready for queries!")

            query = st.text_input("💬 Ask about the document:")
            if query:
                answer, docs = get_answer(query, vectorstore, llm)
                st.markdown("### Answer")
                st.write(answer)

                with st.expander("View relevant sections"):
                    for doc in docs:
                        st.text(doc.page_content[:500] + "...")

        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main()
