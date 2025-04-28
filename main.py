import os
import sys
import pandas as pd
import json
import hashlib
import shutil
from pdf2img import convert_pdf_to_images
from Diagram_extractor import DiagramExtractor
from Fig_desc_extractor import extract_figure_descriptions
import chromadb
from chromadb.utils import embedding_functions

def process_pdf_diagrams(pdf_path, output_base_folder='diagram_extraction_output'):
    """
    Complete pipeline to extract diagrams and their descriptions from a PDF
    and store them in a format suitable for RAG retrieval using ChromaDB
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_base_folder (str): Base folder for storing outputs
    
    Returns:
        pandas.DataFrame: DataFrame containing diagram details and descriptions
    """
    # Create output base folder
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Create subfolder for extracted diagrams
    diagrams_folder = os.path.join(output_base_folder, 'diagrams_store')
    os.makedirs(diagrams_folder, exist_ok=True)
    
    # Create folder for metadata
    metadata_folder = os.path.join(output_base_folder, 'metadata')
    os.makedirs(metadata_folder, exist_ok=True)
    
    # Create folder for vector database
    vector_db_folder = os.path.join(output_base_folder, 'vector_db')
    os.makedirs(vector_db_folder, exist_ok=True)
    
    # Step 1: Convert PDF to images
    pdf_images = convert_pdf_to_images(pdf_path, output_folder=os.path.join(output_base_folder, 'pdf_images'))
    
    # Step 2: Create diagram extractor
    diagram_extractor = DiagramExtractor()
    
    # Lists to store diagram information
    diagram_data = []
    
    # Process each PDF page image
    for page_num, image_path in enumerate(pdf_images, 1):
        # Extract diagrams from the page
        extracted_diagrams = diagram_extractor.extract_diagrams(image_path, output_prefix='diagram')
        
        # Process each extracted diagram
        for diagram_num, diagram_path in enumerate(extracted_diagrams, 1):
            # Extract figure descriptions
            descriptions = extract_figure_descriptions(diagram_path)
            
            # Only add diagrams with descriptions
            if descriptions:
                # Combine all descriptions into a single string
                combined_description = '\n'.join(descriptions)
                
                # Generate a unique ID for the diagram using content hash
                diagram_id = generate_diagram_id(diagram_path, combined_description)
                
                # Copy diagram to permanent storage with the ID as filename
                diagram_extension = os.path.splitext(diagram_path)[1]
                permanent_diagram_path = os.path.join(diagrams_folder, f"{diagram_id}{diagram_extension}")
                shutil.copy(diagram_path, permanent_diagram_path)
                
                # Store diagram information
                diagram_info = {
                    'id': diagram_id,
                    'source_pdf': os.path.basename(pdf_path),
                    'page_number': page_num,
                    'diagram_path': permanent_diagram_path,
                    'description': combined_description
                }
                
                diagram_data.append(diagram_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(diagram_data)
    
    # Only proceed if we have diagrams with descriptions
    if not df.empty:
        # Save structured data
        # 1. Excel for human reading
        excel_path = os.path.join(metadata_folder, 'diagram_extraction_results.xlsx')
        df.to_excel(excel_path, index=False)
        
        # 2. JSON for easy programmatic access
        json_path = os.path.join(metadata_folder, 'diagram_extraction_results.json')
        df.to_json(json_path, orient='records', indent=2)
        
        # 3. Create vector embeddings with ChromaDB
        create_chroma_db(df, vector_db_folder)
        
        print(f"Diagram extraction results saved to:")
        print(f"- Excel: {excel_path}")
        print(f"- JSON: {json_path}")
        print(f"- Vector DB: {vector_db_folder}")
    else:
        print("No diagrams with descriptions found.")
    
    return df

def generate_diagram_id(diagram_path, description):
    """Generate a unique ID for a diagram based on its content and description"""
    # Read the image file in binary mode
    with open(diagram_path, 'rb') as f:
        image_content = f.read()
    
    # Create a hash combining the image content and description
    content_hash = hashlib.md5()
    content_hash.update(image_content)
    content_hash.update(description.encode('utf-8'))
    
    return content_hash.hexdigest()

def create_chroma_db(df, vector_db_folder):
    """Create a ChromaDB collection for diagram descriptions"""
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=vector_db_folder)
    
    # Use sentence-transformers for embeddings
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Good balance of performance and speed
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="diagram_descriptions",
        embedding_function=embedding_function
    )
    
    # Prepare data for ChromaDB
    ids = df['id'].tolist()
    documents = df['description'].tolist()
    metadatas = []
    
    for _, row in df.iterrows():
        metadatas.append({
            "source_pdf": row["source_pdf"],
            "page_number": str(row["page_number"]), 
            "diagram_path": row["diagram_path"],
            "source_page": f"{row['source_pdf']} - Page {row['page_number']}"
        })
    
    # Add documents to collection (this will update existing entries if IDs match)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    # Save collection data separately for backup
    collection_data = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas
    }
    
    with open(os.path.join(vector_db_folder, 'collection_backup.json'), 'w') as f:
        json.dump(collection_data, f, indent=2)

def search_diagrams(query, output_base_folder='diagram_extraction_output', top_k=3):
    """
    Search for diagrams based on a text query using ChromaDB
    
    Args:
        query (str): The search query
        output_base_folder (str): Base folder for stored outputs
        top_k (int): Number of results to return
    
    Returns:
        list: Top matching diagram paths and descriptions
    """
    vector_db_folder = os.path.join(output_base_folder, 'vector_db')
    
    if not os.path.exists(vector_db_folder):
        print("Vector database not found. Falling back to keyword search.")
        return fallback_keyword_search(query, output_base_folder, top_k)
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=vector_db_folder)
        
        # Get collection
        collection = client.get_collection(
            name="diagram_descriptions",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        # Search
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            formatted_results.append({
                'diagram_path': results['metadatas'][0][i]['diagram_path'],
                'description': results['documents'][0][i],
                'source_pdf': results['metadatas'][0][i]['source_pdf'],
                'page_number': results['metadatas'][0][i]['page_number'],
                'relevance_score': results.get('distances', [[0]*len(results['ids'][0])])[0][i]
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"Error searching vector database: {e}")
        return fallback_keyword_search(query, output_base_folder, top_k)

def fallback_keyword_search(query, output_base_folder, top_k=3):
    """Simple keyword-based search as fallback"""
    metadata_path = os.path.join(output_base_folder, 'metadata', 'diagram_extraction_results.json')
    
    if not os.path.exists(metadata_path):
        print("No metadata found.")
        return []
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Simple keyword matching - can be improved with TF-IDF or other methods
    query_terms = query.lower().split()
    
    scored_results = []
    for item in metadata:
        description = item['description'].lower()
        score = sum(1 for term in query_terms if term in description)
        if score > 0:
            scored_results.append((score, item))
    
    # Sort by score (descending)
    scored_results.sort(reverse=True, key=lambda x: x[0])
    
    # Return top_k results
    return [
        {
            'diagram_path': item['diagram_path'],
            'description': item['description'],
            'source_pdf': item['source_pdf'],
            'page_number': item['page_number'],
            'relevance_score': score / len(query_terms)  # Normalize score
        } 
        for score, item in scored_results[:top_k]
    ]

def integrate_with_rag_chatbot(diagram_results):
    """
    Example function showing how to integrate diagram search results with a RAG chatbot
    
    Args:
        diagram_results: Results from search_diagrams function
        
    Returns:
        dict: Data structure for chatbot to use
    """
    if not diagram_results:
        return {
            "found": False,
            "message": "No relevant diagrams found for your query."
        }
    
    # Get the most relevant diagram
    best_match = diagram_results[0]
    
    return {
        "found": True,
        "diagram_path": best_match['diagram_path'],
        "caption": best_match['description'],
        "source": f"From {best_match['source_pdf']}, page {best_match['page_number']}",
        "all_results": diagram_results  # Include all results in case needed
    }

def main():
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_pdf> [--search 'query']")
        sys.exit(1)
    
    # Check if we're in search mode
    if "--search" in sys.argv:
        search_index = sys.argv.index("--search")
        if search_index + 1 < len(sys.argv):
            query = sys.argv[search_index + 1]
            results = search_diagrams(query)
            
            print(f"\n--- Search Results for '{query}' ---")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"- Diagram: {result['diagram_path']}")
                    print(f"- Source: {result['source_pdf']}, Page {result['page_number']}")
                    print(f"- Description: {result['description'][:100]}...")
                    print(f"- Relevance Score: {result.get('relevance_score', 'N/A')}")
            else:
                print("No matching diagrams found.")
        else:
            print("No search query provided.")
    else:
        # Process PDF mode
        pdf_path = sys.argv[1]
        
        # Process PDF
        results = process_pdf_diagrams(pdf_path)
        
        # Print summary
        print("\n--- Extraction Summary ---")
        print(f"Total Diagrams with Descriptions: {len(results)}")
        print("\nTo search for diagrams, run:")
        print(f"python {sys.argv[0]} --search 'your search query'")

if __name__ == '__main__':
    main()