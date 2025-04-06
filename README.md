# 📘 RAG-Based PDF Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF, ask questions about it, and get accurate answers grounded in the document's content. The application uses **LangChain**, **FAISS**, **OpenAI Embeddings**, and **Flan-T5 from Hugging Face**, with a **Streamlit** interface.

---

## 🚀 Features

- Upload any PDF and ask questions about it
- Semantic search via FAISS Vector Store
- Embedding via **OpenAIEmbeddings**
- Answer generation using **Flan-T5-Base**
- Displays both generated answers and reference document chunks

---

## 🛠️ Tech Stack

| Component         | Tool/Library                        |
|------------------|-------------------------------------|
| UI Framework     | Streamlit                           |
| PDF Processing   | PyPDFLoader (LangChain)             |
| Text Splitting   | RecursiveCharacterTextSplitter      |
| Embeddings       | OpenAI Embeddings (`text-embedding-3-small`) |
| Vector DB        | FAISS                               |
| LLM              | Flan-T5-Base (HuggingFace Transformers) |

---

## 📦 Installation

### ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot

2. **Streamlit link:**
     https://rag-chatbot-3qpjmc5vqfuqxu3nrdvy4m.streamlit.app/
