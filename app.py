import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        return splitter.split_documents(pages)
    except Exception as e:
        st.error(f"PDF processing failed: {str(e)}")
        return []

def create_vectorstore(chunks):
    try:
        # Get API key from environment or secrets
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        
        if not api_key:
            st.error("❌ Missing OpenAI API key!")
            raise ValueError("OPENAI_API_KEY not found")

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Validate embeddings
        test_embed = embeddings.embed_query("test")
        if len(test_embed) != 1536:  # OpenAI embedding dimension
            raise ValueError("Embedding validation failed")
            
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Vector store creation failed: {str(e)}")
        raise

def load_llm():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    return ChatOpenAI(
        temperature=0.3,
        model="gpt-3.5-turbo-0125",
        max_retries=3,
        openai_api_key=api_key
    )

def get_answer(query, vectorstore, llm):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer based on the context below. 
        If you don't know the answer, say you don't know.
        
        Context: {context}
        
        Question: {query}
        Answer:"""
        
        response = llm.invoke(prompt)
        return response.content, docs
    except Exception as e:
        st.error(f"Answer generation failed: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="📄 RAG PDF Chatbot", layout="wide")
    st.title("📄 RAG PDF Chatbot")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Analyzing document..."):
                chunks = process_pdf("temp.pdf")
                if not chunks:
                    st.error("No text content found in PDF")
                    return
                
                vectorstore = create_vectorstore(chunks)
                llm = load_llm()

            st.success("Document ready for queries!")
            
            query = st.text_input("Ask about the document:")
            if query:
                answer, docs = get_answer(query, vectorstore, llm)
                st.subheader("Answer")
                st.write(answer)
                
                with st.expander("View source passages"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Passage {i}**")
                        st.text(doc.page_content[:500] + "...")
                        
        except Exception as e:
            st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
