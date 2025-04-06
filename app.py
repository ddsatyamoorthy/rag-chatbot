import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from transformers import pipeline

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
        # Verify API key exists
        if not st.secrets.get("HF_API_TOKEN"):
            st.error("❌ Missing Hugging Face API token in secrets!")
            raise ValueError("Missing HF_API_TOKEN")
            
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=st.secrets["HF_API_TOKEN"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Verify embeddings are working
        test_embed = embeddings.embed_query("test")
        if not isinstance(test_embed, list) or len(test_embed) == 0:
            raise ValueError("Embedding generation failed")
            
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Vector store creation failed: {str(e)}")
        raise

def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device_map="auto",
        torch_dtype="auto"
    )

def get_answer(query, vectorstore, llm):
    try:
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
