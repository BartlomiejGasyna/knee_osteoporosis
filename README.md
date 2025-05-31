# Osteoporosis Classification from Knee X-rays

A deep learning project for classifying osteoporosis severity from knee X-ray images using PyTorch Lightning and EfficientNet.

## Overview

This project implements a computer vision model to automatically classify osteoporosis severity from knee X-ray images into three categories:
- **Normal**: Healthy bone density
- **Osteopenia**: Mild bone density loss (precursor to osteoporosis)
- **Osteoporosis**: Severe bone density loss

## Dataset

The project uses the [Osteoporosis Database](https://www.kaggle.com/datasets/mohamedgobara/osteoporosis-database/data) from Kaggle, which contains knee X-ray images labeled with osteoporosis severity levels.

**Dataset Statistics:**
- Normal: 36 images
- Osteopenia: 154 images  
- Osteoporosis: 49 images
- **Total**: 239 images

Features
- **Automatic balancing**: Uses oversampling to handle class imbalance
- TensorBoard logging

## Acknowledgments

- Dataset: [Mohamed Gobara's Osteoporosis Database](https://www.kaggle.com/datasets/mohamedgobara/osteoporosis-database)
- Model: EfficientNet architecture by Google Research
- Framework: PyTorch Lightning for streamlined training

