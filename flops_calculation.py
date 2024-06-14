# flops_calculation.py
import torch
from thop import profile

def calculate_flops(model, input_size=(1, 3, 1024, 1024), device='cuda'):
    model = model.to(device)
    input_data = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(input_data,))
    print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Total Parameters: {params / 1e6:.2f} MParams")

# Assuming 'my_skin_model' is your model
# from imports import my_skin_model, device
# calculate_flops(my_skin_model)
