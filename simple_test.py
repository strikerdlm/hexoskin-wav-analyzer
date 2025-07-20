import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# Create some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Save a simple plot to verify matplotlib works
plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('Simple Sine Wave')
plt.savefig('test_plot.png')
plt.close()

# Write success message to a file
with open('test_success.txt', 'w') as f:
    f.write("All libraries imported and used successfully!\n")
    f.write(f"NumPy version: {np.__version__}\n")
    f.write(f"Pandas version: {pd.__version__}\n")
    f.write(f"Matplotlib version: {plt.matplotlib.__version__}\n")
    f.write(f"SciPy version: {scipy.__version__}\n") 