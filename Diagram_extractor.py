import os
import cv2
from ultralytics import YOLO

class DiagramExtractor:
    def __init__(self, model_path='runs/detect/train/weights/best.pt', confidence=0.5):
        """
        Initialize Diagram Extractor
        
        Args:
            model_path (str): Path to trained model
            confidence (float): Confidence threshold for detection
        """
        # Create output directories
        self.output_dir = 'extracted_diagrams'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load trained model
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def extract_diagrams(self, image_path, output_prefix='diagram'):
        """
        Extract diagrams from an input image
        
        Args:
            image_path (str): Path to input image
            output_prefix (str): Prefix for output diagram filenames
        
        Returns:
            List of paths to extracted diagram images
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            # Perform detection
            results = self.model(image, conf=self.confidence)
            
            extracted_diagrams = []
            
            # Process each detected diagram
            for i, result in enumerate(results):
                boxes = result.boxes
                
                for j, box in enumerate(boxes):
                    # Extract coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Extract diagram
                    diagram = image[y1:y2, x1:x2]
                    
                    # Generate unique filename
                    filename = f'{output_prefix}_{os.path.basename(image_path)}_{i}_{j}.png'
                    diagram_path = os.path.join(self.output_dir, filename)
                    
                    # Save diagram
                    cv2.imwrite(diagram_path, diagram)
                    extracted_diagrams.append(diagram_path)
                    
                    # Optional: Visualize detection
                    self._draw_detection(image, box)
            
            return extracted_diagrams
        
        except Exception as e:
            print(f"Diagram extraction error: {e}")
            return []
    
    def _draw_detection(self, image, box):
        """
        Draw bounding box on the image
        
        Args:
            image (np.ndarray): Original image
            box (torch.Tensor): Detected box
        """
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def batch_extract(self, input_folder, output_prefix='diagram'):
        """
        Extract diagrams from all images in a folder
        
        Args:
            input_folder (str): Path to folder with images
            output_prefix (str): Prefix for output diagram filenames
        
        Returns:
            List of paths to extracted diagram images
        """
        extracted_diagrams = []
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Process each image in the folder
        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(input_folder, filename)
                diagrams = self.extract_diagrams(image_path, output_prefix)
                extracted_diagrams.extend(diagrams)
        
        return extracted_diagrams

def main():
    # Create extractor instance
    extractor = DiagramExtractor()
    
    # Example: Extract diagrams from a single image
    single_image_path = 'D:\YOLOX\image.png'
    single_image_diagrams = extractor.extract_diagrams(single_image_path)
    print(f"Extracted {len(single_image_diagrams)} diagrams from single image")
    


if __name__ == '__main__':
    main()