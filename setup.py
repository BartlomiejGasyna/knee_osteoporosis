import os
import requests
from tqdm import tqdm
from utils import export_to_onnx

# Define the model checkpoint URL and output path
checkpoint_url = "https://github.com/BartlomiejGasyna/knee_osteoporosis/releases/download/v1.1/osteo_model.ckpt"
checkpoint_path = "models/osteo_model.ckpt"
output_path = "models/osteoporosis_classifier/1/model.onnx"


# Download the model checkpoint


response = requests.get(checkpoint_url, stream=True)
if response.status_code == 200:
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    with open(checkpoint_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        desc='Downloading',
        ascii=True,
        ncols=100,
    ) as bar:
        for data in response.iter_content(chunk_size=block_size):
            if data:  # filter out keep-alive chunks
                f.write(data)
                bar.update(len(data))
    print(f"Model downloaded to {checkpoint_path}")
else:
    print(f"Failed to download model: HTTP {response.status_code}")

# Convert the model to ONNX format
export_to_onnx(checkpoint_path, output_path)
