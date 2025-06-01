
# triton_client.py
"""
Triton Inference Server client for osteoporosis classification
"""

import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import json
import argparse
from utils.preprocessing_utils import OsteoporosisPreprocessor
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
        results['file_path'] = image_path

        return results
    
    def visualize_single_result(self, result, image_path, save_path=None, show=True):
        """
        Visualize single image result with probabilities
        
        Args:
            result: Dictionary containing prediction results
            save_path: Optional path to save the visualization
            show: Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display the X-ray image
        try:
            img = Image.open(image_path)
            ax1.imshow(img, cmap='gray' if img.mode == 'L' else None)
            ax1.set_title(f"X-ray Image\n{Path(image_path).name}", fontsize=12, fontweight='bold')
            ax1.axis('off')
        except Exception as e:
            ax1.text(0.5, 0.5, f"Could not load image:\n{image_path}", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Image Not Available")
        
        # Plot probability distribution
        classes = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        
        # Create color map - highlight predicted class
        colors = ['#ff6b6b' if cls == result['predicted_class'].lower() 
                 else '#4ecdc4' for cls in classes]
        
        bars = ax2.bar(classes, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Osteoporosis Class', fontsize=12, fontweight='bold')
        ax2.set_title(f'Classification Probabilities\nPredicted: {result["predicted_class"].upper()} '
                     f'({result["confidence"]:.1%})', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 0.02,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def predict_dir(self, dir_path):
        """
        Perform batch inference on folder
        
        Args: 
            dir_path: directory path
        Returns:
            List: list of prediction results"""
        if not os.path.isdir(dir_path):
            raise ValueError(f"The directory '{dir_path}' does not exist.")

        results = {}
        for filename in os.listdir(dir_path):
            result = self.predict(os.path.join(dir_path, filename))
            print(f'{filename=}')
            print(f"Confidence: {result['confidence']:.2f}, prediction: {result['predicted_class']}")
            results[filename] = result

        return results


    def visualize_dir_results(self, results_dict, save_path=None, show=True):
        """
        Visualize directory results with summary statistics
        
        Args:
            results_dict: Dictionary with filenames as keys and prediction results as values
            save_path: Optional path to save the visualization
            show: Whether to display the plot
        """
        if not results_dict:
            print("No results to visualize")
            return
        
        # Extract data for visualization
        results = list(results_dict.values())
        classes = list(results[0]['all_probabilities'].keys())
        predicted_classes = [r['predicted_class'].lower() for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Class distribution pie chart
        class_counts = {cls: predicted_classes.count(cls) for cls in classes}
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        wedges, texts, autotexts = ax1.pie(class_counts.values(), labels=class_counts.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title(f'Predicted Class Distribution\n({len(results)} images)', fontsize=12, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        # 2. Confidence distribution histogram
        ax2.hist(confidences, bins=min(15, len(confidences)), alpha=0.7, color='skyblue', 
                edgecolor='black', linewidth=1)
        ax2.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(confidences):.1%}')
        ax2.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax2.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total images: {len(results)}")
        print(f"Average confidence: {np.mean(confidences):.1%}")
        print(f"Confidence range: {np.min(confidences):.1%} - {np.max(confidences):.1%}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

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
    parser.add_argument('--image', type=str, default=None, help='Path to X-ray image')
    parser.add_argument('--dir', type=str, help='Path to X-ray folder')
    parser.add_argument('--server', type=str, default='localhost:8000', help='Triton server URL')
    parser.add_argument('--visualize', action='store_true', help='Show visualization of results')
    parser.add_argument('--save-viz', type=str, help='Save visualization to specified path')
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
        
        if args.image is not None:
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
        
            # Visualization for single image
            if args.visualize or args.save_viz:
                client.visualize_single_result(
                    results,
                    image_path=args.image,
                    save_path=args.save_viz,
                    show=args.visualize
                )

        elif args.dir is not None:
            # Perform inference
            print(f"\nAnalyzing directory: {args.dir}")
            results = client.predict_dir(args.dir)

            if args.visualize or args.save_viz:
                client.visualize_dir_results(
                    results,
                    save_path=args.save_viz,
                    show=args.visualize
                )
        else:
            print('No image path or directory for inference provided!') 

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()