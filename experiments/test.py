import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import model_inference

# generate sample data and display it
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.show()
