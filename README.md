# Diagram Extraction and Semantic Search Pipeline

This project provides a pipeline to extract diagrams from PDF files, generate their descriptions, and enable semantic search over the extracted diagrams using ChromaDB. It is designed for use in document understanding and Retrieval-Augmented Generation (RAG) chatbot applications.

---

## Features

- **PDF to Image Conversion:** Converts each page of a PDF into images.
- **Diagram Detection:** Uses a YOLO-based model to detect and extract diagrams from images.
- **Figure Description Extraction:** Extracts textual descriptions for each diagram.
- **Metadata Storage:** Stores results in Excel and JSON formats.
- **Semantic Search:** Indexes diagram descriptions in ChromaDB for fast, semantic search.
- **Fallback Keyword Search:** If vector search is unavailable, falls back to simple keyword matching.
- **RAG Chatbot Integration:** Provides a function to integrate search results with a chatbot.

---

## Project Structure

├── main.py # Main pipeline and CLI
├── pdf2img.py # PDF to image conversion
├── Diagram_extractor.py # Diagram extraction using YOLO
├── Fig_desc_extractor.py # Figure description extraction
├── environment.yml # Conda environment with dependencies
├── diagram_extraction_output/ # Output folder (created after running)
│ ├── diagrams_store/ # Extracted diagrams
│ ├── metadata/ # Excel/JSON metadata
│ └── vector_db/ # ChromaDB vector database

