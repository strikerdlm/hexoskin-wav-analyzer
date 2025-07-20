import numpy as np
import pandas as pd
import matplotlib
import scipy

# Write versions to a file
with open('library_versions.txt', 'w') as f:
    f.write("NumPy version: " + np.__version__ + "\n")
    f.write("Pandas version: " + pd.__version__ + "\n")
    f.write("Matplotlib version: " + matplotlib.__version__ + "\n")
    f.write("SciPy version: " + scipy.__version__ + "\n")
    f.write("\nAll libraries imported successfully!") 