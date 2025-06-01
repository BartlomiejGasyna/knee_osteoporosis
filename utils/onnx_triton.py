from .classifier import OsteoporosisClassifier, IMG_SIZE
import torch
import onnx
import onnxruntime
import numpy as np
import os

def export_to_onnx(checkpoint_path, output_path=None):
    """
    Export PyTorch Lightning model to ONNX format for Triton
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "models/osteoporosis_classifier/1/model.onnx")
    
    # Load the trained model
    model = OsteoporosisClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=3,
        learning_rate=0.001
    )
    
    # Set model to evaluation mode
    model.eval()
    
    device = torch.device("cuda")    

    model = model.to(device)
    # Create dummy input with correct shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, *IMG_SIZE).to(device)


    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Compatible with most deployment scenarios
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable batch size
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification successful!")
    
    # Test with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(output_path)
    test_input = np.random.randn(1, 3, *IMG_SIZE).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"ONNX Runtime test successful! Output shape: {ort_outs[0].shape}")
    
    return output_path


def create_triton_config():
    """
    Create Triton model configuration
    """
    config = {
        "name": "osteoporosis_classifier",
        "platform": "onnxruntime_onnx",
        "max_batch_size": 8,
        "input": [
            {
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [3, *IMG_SIZE]
            }
        ],
        "output": [
            {
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": [3]
            }
        ],
        "dynamic_batching": {
            "preferred_batch_size": [1, 2, 4],
            "max_queue_delay_microseconds": 100
        },
        "instance_group": [
            {
                "count": 1,
                "kind": "KIND_GPU"
            }
        ],
        "optimization": {
            "execution_accelerators": {
                "gpu_execution_accelerator": [
                    {
                        "name": "tensorrt",
                        "parameters": {
                            "precision_mode": "FP16",
                            "max_workspace_size_bytes": "1073741824"
                        }
                    }
                ]
            }
        }
    }
    
    # Create config file
    config_path = "models/osteoporosis_classifier/config.pbtxt"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Convert to protobuf text format
    config_content = f'''name: "{config['name']}"
platform: "{config['platform']}"
max_batch_size: {config['max_batch_size']}

input [
  {{
    name: "{config['input'][0]['name']}"
    data_type: {config['input'][0]['data_type']}
    dims: [ {', '.join(map(str, config['input'][0]['dims']))} ]
  }}
]

output [
  {{
    name: "{config['output'][0]['name']}"
    data_type: {config['output'][0]['data_type']}
    dims: [ {config['output'][0]['dims'][0]} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {', '.join(map(str, config['dynamic_batching']['preferred_batch_size']))} ]
  max_queue_delay_microseconds: {config['dynamic_batching']['max_queue_delay_microseconds']}
}}

instance_group [
  {{
    count: {config['instance_group'][0]['count']}
    kind: {config['instance_group'][0]['kind']}
  }}
]

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator [
      {{
        name: "{config['optimization']['execution_accelerators']['gpu_execution_accelerator'][0]['name']}"
        parameters [
          {{
            key: "precision_mode"
            value: "{config['optimization']['execution_accelerators']['gpu_execution_accelerator'][0]['parameters']['precision_mode']}"
          }},
          {{
            key: "max_workspace_size_bytes"
            value: "{config['optimization']['execution_accelerators']['gpu_execution_accelerator'][0]['parameters']['max_workspace_size_bytes']}"
          }}
        ]
      }}
    ]
  }}
}}'''
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Triton config created at {config_path}")
    return config_path