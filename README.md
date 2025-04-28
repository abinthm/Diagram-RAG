# Diagram Extraction and Semantic Search Pipeline

A comprehensive solution for extracting diagrams from PDF documents, capturing their descriptions, and enabling intelligent semantic search using vector embeddings.


## 🚀 Features

- **📄 PDF Processing**: Convert PDFs to high-quality images for analysis
- **🔍 Diagram Detection**: Extract diagrams using state-of-the-art YOLO object detection
- **📝 OCR Description Extraction**: Automatically capture figure captions and descriptions
- **🧠 Vector Search**: Index diagrams with semantic embeddings for intelligent retrieval
- **📊 Structured Metadata**: Export results in easily consumable Excel and JSON formats
- **🤖 RAG Integration**: Ready-to-use functions for integration with RAG chatbots

## 📋 Project Structure

```
├── main.py                       # Main pipeline and CLI interface
├── pdf2img.py                    # PDF to image conversion utilities
├── Diagram_extractor.py          # YOLO-based diagram extraction
├── Fig_desc_extractor.py         # OCR figure description extraction
├── environment.yml               # Conda environment configuration
└── diagram_extraction_output/    # Generated output folder
    ├── diagrams_store/           # Extracted diagram images
    ├── metadata/                 # Excel and JSON metadata
    └── vector_db/                # ChromaDB vector database
```

## 🛠️ Installation

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/diagram-extraction-pipeline.git
   cd diagram-extraction-pipeline
   ```

2. Create and activate the conda environment
   ```bash
   conda env create -f environment.yml
   conda activate diagram_extractor
   ```

3. Install Tesseract OCR (required for text extraction)
   - Windows: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

## 💻 Usage

### Extract diagrams from a PDF

```bash
python main.py path/to/your/document.pdf
```

### Search for diagrams by query

```bash
python main.py --search "neural network architecture"
```

### Programmatic Usage

```python
from main import process_pdf_diagrams, search_diagrams

# Process a PDF
results_df = process_pdf_diagrams("path/to/document.pdf")

# Search for diagrams
diagram_results = search_diagrams("Bluetooth data frame")

# Integrate with a RAG chatbot
from main import integrate_with_rag_chatbot
rag_data = integrate_with_rag_chatbot(diagram_results)
```

## 📊 Example Output

After processing a PDF, the system generates:

1. **Extracted Diagrams**: PNG images in the `diagrams_store` folder
2. **Metadata Excel File**: Complete listing with diagram IDs and descriptions
3. **Metadata JSON File**: Machine-readable format for programmatic access
4. **Vector Database**: ChromaDB collection with embeddings for semantic search

## 🧪 How It Works

1. **PDF Processing**: Convert PDF pages to images using PyMuPDF
2. **Diagram Detection**: Detect diagrams using a trained YOLO model
3. **Text Extraction**: Extract figure descriptions using Tesseract OCR
4. **Embedding Generation**: Create vector embeddings using sentence-transformers
5. **Search**: Query diagrams by semantic similarity or keywords
