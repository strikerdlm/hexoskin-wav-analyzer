try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy
    
    with open('library_test_results.txt', 'w') as f:
        f.write(f"NumPy version: {np.__version__}\n")
        f.write(f"Pandas version: {pd.__version__}\n")
        f.write(f"Matplotlib version: {plt.matplotlib.__version__}\n")
        f.write(f"SciPy version: {scipy.__version__}\n")
        f.write("\nAll libraries imported successfully!")
except Exception as e:
    with open('library_test_results.txt', 'w') as f:
        f.write(f"Error: {str(e)}") 