import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class OsteoporosisPreprocessor:
    """Preprocessing for osteoporosis classification"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.class_names = ['normal', 'osteopenia', 'osteoporosis']
    
    def preprocess_image(self, image_path_or_pil):
        """
        Preprocess image for inference
        
        Args:
            image_path_or_pil: Path to image file or PIL Image
            
        Returns:
            np.ndarray: Preprocessed image ready for inference
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension and convert to numpy
        batch = tensor.unsqueeze(0).numpy()
        
        return batch
    
    def postprocess_output(self, logits):
        """
        Convert model output to human-readable predictions
        
        Args:
            logits: Raw model output
            
        Returns:
            dict: Prediction results
        """
        # Apply softmax to get probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities, axis=1)[0]
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx]
        
        # Create detailed results
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                class_name: float(prob) for class_name, prob in 
                zip(self.class_names, probabilities[0])
            }
        }
        
        return results

# Example usage:
# preprocessor = OsteoporosisPreprocessor()
# input_data = preprocessor.preprocess_image("path/to/xray.jpg")
# # Send input_data to Triton server
# # results = preprocessor.postprocess_output(server_response)