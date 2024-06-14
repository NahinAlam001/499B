# Project

## Description
This project contains scripts for loading and preprocessing data, training and evaluating a model, and calculating FLOPs for a segmentation model.

## Structure
- `imports.py`: Contains all necessary imports.
- `data_loading.py`: Script for loading and preprocessing data.
- `evaluation.py`: Script for evaluating the model.
- `flops_calculation.py`: Script for calculating FLOPs.
- `train.py`: Script for training the model.

## Usage
1. **Data Loading**
   ```bash
   python data_loading.py

2. **Model Training**
   ```bash
   python train.py

3. **Model Evaluation**
   ```bash
   from imports import my_skin_model, dataset, device
   from evaluation import evaluate
   evaluate(my_skin_model, dataset, device)

4. **FLOPs Calculation**
  ```bash
  from imports import my_skin_model, device
  from flops_calculation import calculate_flops
  calculate_flops(my_skin_model)
