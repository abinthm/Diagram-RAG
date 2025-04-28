import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import re

def extract_figure_descriptions(image_path, max_length=5000):
    """
    Extract comprehensive figure descriptions with advanced handling
    
    Args:
        image_path (str): Path to the image file
        max_length (int): Maximum length of text to extract
    
    Returns:
        list: Cleaned figure descriptions
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Use Tesseract to do OCR on the image
        full_text = pytesseract.image_to_string(image)[:max_length]
        
        # Advanced figure description extraction patterns
        figure_patterns = [
            # Pattern to capture Figure 1:, Figure 5.1, etc. with multi-line descriptions
            r'(Figure\s*(?:\d+(?:\.\d+)?)\s*(?:\((?:\w)\))?\s*:?.*?(?:\n\s*.*?)*?(?=\n\n|\n(?:Figure|\Z)))',
            
            # Alternative comprehensive pattern
            r'(Figure\s*\d+(?:\.\d+)?.*?(?:\n\s*.*?)*?(?=\n\n|\n(?:Figure|\Z)))'
        ]
        
        # Comprehensive extraction attempts
        for pattern in figure_patterns:
            figure_matches = re.findall(pattern, full_text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            
            if figure_matches:
                # Advanced cleaning and normalization
                cleaned_matches = []
                for match in figure_matches:
                    # Remove excessive whitespace, normalize line breaks
                    cleaned_match = re.sub(r'\s+', ' ', match.strip())
                    # Remove multiple spaces
                    cleaned_match = re.sub(r'\s{2,}', ' ', cleaned_match)
                    cleaned_matches.append(cleaned_match)
                
                return cleaned_matches
        
        # Fallback if no matches found
        return []
    
    except Exception as e:
        print(f"Error extracting text: {e}")
        return []

def validate_extraction(image_path):
    """
    Validate and demonstrate figure description extraction
    
    Args:
        image_path (str): Path to the image file
    """
    figures = extract_figure_descriptions(image_path)
    
    if figures:
        print("\nExtracted Figure Descriptions:")
        for i, fig in enumerate(figures, 1):
            print(f"Extracted Figure {i}:")
            print(fig)
            print("-" * 50)
    else:
        print("No figure descriptions found.")
        
# Example usage
if __name__ == "__main__":
    image_path = 'D:\YOLOX\extracted_diagrams\diagram_image.png_0_0.png'
    validate_extraction(image_path)