import os
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path

def convert_pdf_to_images(pdf_path, output_folder=None, dpi=300):
    """
    Convert each page of a PDF to an image using PyMuPDF.
    
    Args:
    pdf_path (str): Path to the input PDF file
    output_folder (str, optional): Path to the output folder for images. 
                                   If not provided, creates a folder next to the PDF.
    dpi (int, optional): Resolution of the output images. Default is 300.
    
    Returns:
    list: Paths to the generated image files
    """
    # Validate PDF path
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file {pdf_path} does not exist.")
    
    # Determine output folder
    if output_folder is None:
        # Create a folder with the same name as the PDF in the same directory
        pdf_name = Path(pdf_path).stem
        output_folder = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_images")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert PDF to images
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Store image paths
        image_paths = []
        
        # Iterate through pages
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]
            
            # Render page to an image
            # The matrix argument scales the image (1.0 = 72 dpi)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Generate filename (page_001.jpg, page_002.jpg, etc.)
            filename = os.path.join(output_folder, f"page4_{page_num+1:03d}.jpg")
            
            # Save the image
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(filename, "JPEG")
            image_paths.append(filename)
        
        # Close the PDF document
        pdf_document.close()
        
        print(f"Successfully converted {len(image_paths)} pages to images in {output_folder}")
        return image_paths
    
    except Exception as e:
        print(f"An error occurred during PDF conversion: {e}")
        return []

def main():
    # Example usage
    pdf_path = "AI Textbook.pdf"
    
    # Optional: Specify a custom output folder
    output_folder = "CN"
    output_folder = output_folder if output_folder else None
    
    # Optional: Specify DPI
    try:
        dpi = int(input("Enter DPI for image resolution (default is 300): ") or 300)
    except ValueError:
        dpi = 300
    
    # Convert PDF to images
    converted_images = convert_pdf_to_images(pdf_path, output_folder, dpi)
    
    # Print paths of converted images
    if converted_images:
        print("\nConverted Image Paths:")
        for img_path in converted_images:
            print(img_path)

if __name__ == "__main__":
    main()