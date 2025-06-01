# Dockerfile
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install additional Python packages if needed
RUN pip install pillow numpy torch torchvision pytorch-lightning onnx onnxruntime-gpu

# Copy model repository
COPY models/ /models/

# Set model repository path
ENV TRITON_MODEL_REPOSITORY=/models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--allow-gpu-metrics=false"]
