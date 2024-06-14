import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from scipy import ndimage
import imageio
import torch
from datasets import Dataset
from sklearn.metrics import jaccard_score, confusion_matrix

!pip install datasets
!pip install -q monai
try:
    from thop import profile
except:
    !pip install thop
    from thop import profile
