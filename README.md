# 📘 RAG-Based PDF Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF, ask questions about it, and get accurate answers grounded in the document's content. The application uses **LangChain**, **FAISS**, **Sentence Transformers**, and **Hugging Face Transformers**, with a **Streamlit** interface.

---

## 🚀 Features

- Upload any PDF and ask questions about it
- Uses semantic search (FAISS) for context retrieval
- Generates answers using `flan-t5-base` from HuggingFace
- Displays retrieved context and matched document chunks for debugging

---

## 🛠️ Tech Stack

| Component         | Tool/Library                        |
|------------------|-------------------------------------|
| UI Framework     | Streamlit                           |
| PDF Processing   | PyPDFLoader (LangChain)             |
| Text Splitting   | RecursiveCharacterTextSplitter      |
| Embeddings       | Sentence Transformers (`MiniLM`)    |
| Vector DB        | FAISS                               |
| LLM              | Flan-T5-Base (HuggingFace)          |

---

## 📦 Installation

### ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
2. **Streamlit link:**
     https://rag-chatbot-3qpjmc5vqfuqxu3nrdvy4m.streamlit.app/
