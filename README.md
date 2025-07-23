# 🧠 Multimodal RAG using Qwen-0.6B and Colpali BiQwen2.5

This project implements a lightweight **Multimodal Retrieval-Augmented Generation (RAG)** pipeline for answering questions based on the visual content of PDF documents.

## 🔍 What This Project Does

- Converts a PDF into page-wise images  
- Encodes the image representations using **Colpali BiQwen2.5**, a vision-language embedding model  
- Stores these embeddings in a **FAISS** vector database  
- Retrieves relevant image-based chunks in response to user queries  
- Uses **Qwen/Qwen3-0.6B**, a compact open-source LLM, to generate answers from the retrieved content

## 📌 Key Features

- ✅ Fully multimodal: No OCR or text extraction needed  
- 🚀 Lightweight and Colab-friendly  
- 🔌 Compatible with any kind of PDF document  
- 🤖 Natural question-answering using LangChain and HuggingFace models

## 🧰 Tech Stack

- `Colpali BiQwen2.5` – Vision-language embedding model  
- `FAISS` – Vector search engine  
- `LangChain` – RAG pipeline management  
- `Transformers` – Model hub integration  
- `pdf2image` + `Poppler` – PDF to image conversion  
- `Qwen3-0.6B` – Language model for answer generation

## 💡 Use Cases

- Visual-first analysis of annual reports  
- Compliance and legal document Q&A  
- OCR-free document search  
- Educational PDF understanding  
- Multimodal document automation

## 📄 File Instructions

Upload your PDF (e.g., `annual_report.pdf`) into the project directory. The script will automatically:

1. Convert each page into an image  
2. Encode those images as document chunks  
3. Embed and index them using Colpali  
4. Enable querying using a natural language prompt

## ⚠️ Limitations

- Intended for prototyping and demonstration purposes

## 🚀 Future Improvements

- Integrate true image encoding with vision-language models  
- Combine text + image hybrid chunking  
- Add document classification or summarization modules  
- Expand to support live webcam-based document intake

## 📜 License

This project is licensed under the **MIT License**.  

