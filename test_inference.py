
# test_inference.py
"""
Comprehensive testing script for the deployed model
"""

import os
import time
import numpy as np
from triton_client import TritonOsteoporosisClient
import matplotlib.pyplot as plt
from PIL import Image

def performance_test(client, image_path, num_iterations=100):
    """Test inference performance"""
    print(f"Running performance test with {num_iterations} iterations...")
    
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        _ = client.predict(image_path)
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    times = np.array(times)
    
    print(f"\nPerformance Results:")
    print(f"Mean inference time: {times.mean():.3f}s")
    print(f"Std deviation: {times.std():.3f}s")
    print(f"Min time: {times.min():.3f}s")
    print(f"Max time: {times.max():.3f}s")
    print(f"95th percentile: {np.percentile(times, 95):.3f}s")
    print(f"Throughput: {1.0/times.mean():.1f} images/second")

def batch_test(client, image_paths):
    """Test batch inference"""
    print(f"Testing batch inference with {len(image_paths)} images...")
    
    start_time = time.time()
    results = client.predict_batch(image_paths)
    end_time = time.time()
    
    batch_time = end_time - start_time
    throughput = len(image_paths) / batch_time
    
    print(f"Batch inference time: {batch_time:.3f}s")
    print(f"Batch throughput: {throughput:.1f} images/second")
    
    return results

def visualize_results(image_path, results):
    """Visualize prediction results"""
    img = Image.open(image_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(img, cmap='gray' if img.mode == 'L' else None)
    ax1.set_title('X-ray Image')
    ax1.axis('off')
    
    # Show predictions
    classes = list(results['all_probabilities'].keys())
    probs = list(results['all_probabilities'].values())
    
    colors = ['green' if cls == results['predicted_class'] else 'lightblue' for cls in classes]
    bars = ax2.bar(classes, probs, color=colors)
    ax2.set_title(f'Prediction: {results["predicted_class"].upper()}\nConfidence: {results["confidence"]:.2%}')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main testing function"""
    # Initialize client
    try:
        client = TritonOsteoporosisClient()
        print("✓ Client connected successfully")
    except Exception as e:
        print(f"✗ Failed to connect to Triton server: {e}")
        return
    
    # Test with a sample image (you'll need to provide a real image path)
    test_image = "/home/gasyna/.cache/kagglehub/datasets/mohamedgobara/osteoporosis-database/versions/1/Osteoporosis Knee X-ray/normal/N13.jpg"  
    # test_image = "/home/gasyna/.cache/kagglehub/datasets/mohamedgobara/osteoporosis-database/versions/1/Osteoporosis Knee X-ray/osteoporosis/OS17.jpg"
    # test_image = "N14.jpg"
    # test_image = "N12.JPEG"
    if not os.path.exists(test_image):
        print(f"Please provide a valid test image path. Looking for: {test_image}")
        return
    
    print(f"\nTesting with image: {test_image}")
    
    # Single inference test
    try:
        results = client.predict(test_image)
        print("✓ Single inference successful")
        print(f"  Prediction: {results['predicted_class']}")
        print(f"  Confidence: {results['confidence']:.2%}")
    except Exception as e:
        print(f"✗ Single inference failed: {e}")
        return
    
    # Performance test
    try:
        performance_test(client, test_image, num_iterations=50)
        print("✓ Performance test completed")
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
    
    # Batch test (if you have multiple images)
    test_images = [test_image] * 4  # Using same image for demo
    try:
        batch_results = batch_test(client, test_images)
        print("✓ Batch inference successful")
    except Exception as e:
        print(f"✗ Batch inference failed: {e}")
    
    # Visualization (optional)
    try:
        visualize_results(test_image, results)
        print("✓ Visualization completed")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

if __name__ == "__main__":
    main()