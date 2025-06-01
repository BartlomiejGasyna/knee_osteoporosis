
# triton_client.py
"""
Triton Inference Server client for osteoporosis classification
"""

import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import json
import argparse
from preprocessing_utils import OsteoporosisPreprocessor

class TritonOsteoporosisClient:
    """Client for osteoporosis classification using Triton"""
    
    def __init__(self, server_url="localhost:8000", model_name="osteoporosis_classifier"):
        self.server_url = server_url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=server_url)
        self.preprocessor = OsteoporosisPreprocessor()
        
        # Check if server is ready
        if not self.client.is_server_ready():
            raise Exception(f"Triton server at {server_url} is not ready")
        
        # Check if model is ready
        if not self.client.is_model_ready(model_name):
            raise Exception(f"Model {model_name} is not ready")
        
        print(f"Connected to Triton server at {server_url}")
        print(f"Model {model_name} is ready for inference")
    
    def get_model_metadata(self):
        """Get model metadata"""
        return self.client.get_model_metadata(self.model_name)
    
    def predict(self, image_path):
        """
        Perform inference on a single image
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        input_data = self.preprocessor.preprocess_image(image_path)
        
        # Create Triton input
        inputs = []
        inputs.append(httpclient.InferInput("input", input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Create Triton output
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("output"))
        
        # Send inference request
        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        # Get result
        logits = response.as_numpy("output")
        
        # Postprocess
        results = self.preprocessor.postprocess_output(logits)
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Perform batch inference on multiple images
        
        Args:
            image_paths: List of paths to X-ray images
            
        Returns:
            list: List of prediction results
        """
        # Preprocess all images
        batch_data = []
        for image_path in image_paths:
            input_data = self.preprocessor.preprocess_image(image_path)
            batch_data.append(input_data[0])  # Remove single batch dimension
        
        # Stack into batch
        batch_input = np.stack(batch_data, axis=0)
        
        # Create Triton input
        inputs = []
        inputs.append(httpclient.InferInput("input", batch_input.shape, "FP32"))
        inputs[0].set_data_from_numpy(batch_input)
        
        # Create Triton output
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("output"))
        
        # Send inference request
        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        # Get results
        batch_logits = response.as_numpy("output")
        
        # Postprocess each result
        results = []
        for i, logits in enumerate(batch_logits):
            result = self.preprocessor.postprocess_output(logits.reshape(1, -1))
            result['image_path'] = image_paths[i]
            results.append(result)
        
        return results

def main():
    """Example usage of the Triton client"""
    parser = argparse.ArgumentParser(description='Triton Osteoporosis Classification Client')
    parser.add_argument('--image', type=str, required=True, help='Path to X-ray image')
    parser.add_argument('--server', type=str, default='localhost:8000', help='Triton server URL')
    parser.add_argument('--model', type=str, default='osteoporosis_classifier', help='Model name')
    
    args = parser.parse_args()
    
    try:
        # Create client
        client = TritonOsteoporosisClient(
            server_url=args.server,
            model_name=args.model
        )
        
        # Get model info
        metadata = client.get_model_metadata()
        print(f"Model version: {metadata['versions']}")
        
        # Perform inference
        print(f"\nAnalyzing image: {args.image}")
        results = client.predict(args.image)
        
        # Display results
        print("\n" + "="*50)
        print("OSTEOPOROSIS CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Predicted Class: {results['predicted_class'].upper()}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nDetailed Probabilities:")
        for class_name, prob in results['all_probabilities'].items():
            print(f"  {class_name.capitalize()}: {prob:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()